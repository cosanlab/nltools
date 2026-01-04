"""Tests for pipeline terminals and results (Phase 3).

Tests cover:
- FoldResult: per-fold result container
- CVResult: aggregated CV results
- PredictTerminal: prediction algorithms and evaluation
"""

import numpy as np
import pytest

from nltools.pipelines.base import FittedStack
from nltools.pipelines.results import CVResult, FoldResult
from nltools.pipelines.terminals import PredictTerminal


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_creation(self):
        """Test FoldResult creation."""
        result = FoldResult(
            score=0.85,
            predictions=np.array([1, 2, 3]),
            train_idx=np.array([0, 1, 2]),
            test_idx=np.array([3, 4, 5]),
            fitted_stack=FittedStack(),
        )
        assert result.score == 0.85
        assert len(result.predictions) == 3
        assert len(result.train_idx) == 3
        assert len(result.test_idx) == 3

    def test_repr(self):
        """Test string representation."""
        result = FoldResult(
            score=0.75,
            predictions=np.array([1, 2]),
            train_idx=np.array([0, 1]),
            test_idx=np.array([2, 3]),
            fitted_stack=FittedStack(),
        )
        r = repr(result)
        assert "0.75" in r
        assert "n_test=2" in r


class TestCVResult:
    """Tests for CVResult aggregation."""

    @pytest.fixture
    def sample_fold_results(self):
        """Create sample fold results for testing."""
        return [
            FoldResult(
                score=0.8,
                predictions=np.array([1.0, 2.0, 3.0]),
                train_idx=np.array([3, 4, 5, 6, 7, 8]),
                test_idx=np.array([0, 1, 2]),
                fitted_stack=FittedStack(),
            ),
            FoldResult(
                score=0.7,
                predictions=np.array([4.0, 5.0, 6.0]),
                train_idx=np.array([0, 1, 2, 6, 7, 8]),
                test_idx=np.array([3, 4, 5]),
                fitted_stack=FittedStack(),
            ),
            FoldResult(
                score=0.9,
                predictions=np.array([7.0, 8.0, 9.0]),
                train_idx=np.array([0, 1, 2, 3, 4, 5]),
                test_idx=np.array([6, 7, 8]),
                fitted_stack=FittedStack(),
            ),
        ]

    def test_scores(self, sample_fold_results):
        """Test scores property."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        np.testing.assert_array_almost_equal(result.scores, [0.8, 0.7, 0.9])

    def test_mean_score(self, sample_fold_results):
        """Test mean_score property."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        assert result.mean_score == pytest.approx(0.8, abs=0.01)

    def test_std_score(self, sample_fold_results):
        """Test std_score property."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        expected_std = np.std([0.8, 0.7, 0.9])
        assert result.std_score == pytest.approx(expected_std, abs=0.001)

    def test_n_folds(self, sample_fold_results):
        """Test n_folds property."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        assert result.n_folds == 3

    def test_predictions_reconstruction(self, sample_fold_results):
        """Test predictions are reconstructed in original order."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        preds = result.predictions

        # Should have 9 samples total
        assert preds.shape == (9,)

        # Check each fold's predictions are in correct positions
        np.testing.assert_array_equal(preds[0:3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(preds[3:6], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(preds[6:9], [7.0, 8.0, 9.0])

    def test_predictions_multioutput(self):
        """Test predictions work with multi-output."""
        fold_results = [
            FoldResult(
                score=0.8,
                predictions=np.array([[1, 2], [3, 4]]),
                train_idx=np.array([2, 3]),
                test_idx=np.array([0, 1]),
                fitted_stack=FittedStack(),
            ),
            FoldResult(
                score=0.7,
                predictions=np.array([[5, 6], [7, 8]]),
                train_idx=np.array([0, 1]),
                test_idx=np.array([2, 3]),
                fitted_stack=FittedStack(),
            ),
        ]
        result = CVResult(fold_results=fold_results, pipeline=None)
        preds = result.predictions

        assert preds.shape == (4, 2)
        np.testing.assert_array_equal(preds[0], [1, 2])
        np.testing.assert_array_equal(preds[2], [5, 6])

    def test_is_fully_invertible_empty(self):
        """Test invertibility with empty results."""
        result = CVResult(fold_results=[], pipeline=None)
        assert result.is_fully_invertible is True

    def test_summary(self, sample_fold_results):
        """Test summary string."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        summary = result.summary()

        assert "3 folds" in summary
        assert "Mean score" in summary
        assert "0.8" in summary

    def test_repr(self, sample_fold_results):
        """Test string representation."""
        result = CVResult(fold_results=sample_fold_results, pipeline=None)
        r = repr(result)

        assert "CVResult" in r
        assert "n_folds=3" in r


class TestPredictTerminal:
    """Tests for PredictTerminal prediction."""

    @pytest.fixture
    def regression_data(self):
        """Create regression test data."""
        np.random.seed(42)
        n_train, n_test = 50, 20
        n_features = 10

        X_train = np.random.randn(n_train, n_features)
        X_test = np.random.randn(n_test, n_features)

        # Create target with some signal
        true_weights = np.random.randn(n_features)
        y_train = X_train @ true_weights + np.random.randn(n_train) * 0.1
        y_test = X_test @ true_weights + np.random.randn(n_test) * 0.1
        y = np.concatenate([y_train, y_test])

        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_train + n_test)

        return X_train, X_test, y, train_idx, test_idx

    @pytest.fixture
    def classification_data(self):
        """Create classification test data."""
        np.random.seed(42)
        n_train, n_test = 50, 20
        n_features = 10

        # Create separable classes
        X_train = np.vstack(
            [
                np.random.randn(n_train // 2, n_features) + 2,
                np.random.randn(n_train // 2, n_features) - 2,
            ]
        )
        X_test = np.vstack(
            [
                np.random.randn(n_test // 2, n_features) + 2,
                np.random.randn(n_test // 2, n_features) - 2,
            ]
        )

        y_train = np.array([0] * (n_train // 2) + [1] * (n_train // 2))
        y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))
        y = np.concatenate([y_train, y_test])

        train_idx = np.arange(n_train)
        test_idx = np.arange(n_train, n_train + n_test)

        return X_train, X_test, y, train_idx, test_idx

    def test_ridge_regression(self, regression_data):
        """Test ridge regression."""
        X_train, X_test, y, train_idx, test_idx = regression_data

        terminal = PredictTerminal(y=y, algorithm="ridge")
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)
        assert result.score > 0.5  # Should have good fit with signal
        assert len(result.predictions) == len(test_idx)

    def test_lasso_regression(self, regression_data):
        """Test lasso regression."""
        X_train, X_test, y, train_idx, test_idx = regression_data

        terminal = PredictTerminal(y=y, algorithm="lasso", kwargs={"alpha": 0.1})
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)
        assert len(result.predictions) == len(test_idx)

    def test_svr(self, regression_data):
        """Test SVR."""
        X_train, X_test, y, train_idx, test_idx = regression_data

        terminal = PredictTerminal(y=y, algorithm="svr")
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)

    def test_svm_classification(self, classification_data):
        """Test SVM classification."""
        X_train, X_test, y, train_idx, test_idx = classification_data

        terminal = PredictTerminal(y=y, algorithm="svm")
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)
        assert result.score > 0.8  # Should classify well-separated data

    def test_logistic_regression(self, classification_data):
        """Test logistic regression."""
        X_train, X_test, y, train_idx, test_idx = classification_data

        terminal = PredictTerminal(y=y, algorithm="logistic")
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)
        assert result.score > 0.8

    def test_random_forest_classification(self, classification_data):
        """Test random forest auto-detects classification."""
        X_train, X_test, y, train_idx, test_idx = classification_data

        terminal = PredictTerminal(
            y=y, algorithm="rf", kwargs={"n_estimators": 10, "random_state": 42}
        )
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)
        assert result.score > 0.7

    def test_random_forest_regression(self, regression_data):
        """Test random forest auto-detects regression."""
        X_train, X_test, y, train_idx, test_idx = regression_data

        terminal = PredictTerminal(
            y=y, algorithm="rf", kwargs={"n_estimators": 10, "random_state": 42}
        )
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        assert isinstance(result, FoldResult)

    def test_unknown_algorithm_raises(self):
        """Test unknown algorithm raises error."""
        terminal = PredictTerminal(y=np.array([1, 2, 3]), algorithm="unknown")

        with pytest.raises(ValueError, match="Unknown algorithm"):
            terminal._get_model()

    def test_with_y(self, regression_data):
        """Test with_y creates new terminal."""
        X_train, X_test, y, train_idx, test_idx = regression_data

        terminal1 = PredictTerminal(y=y, algorithm="ridge")
        new_y = np.random.randn(len(y))
        terminal2 = terminal1.with_y(new_y)

        assert terminal1 is not terminal2
        assert terminal2.algorithm == "ridge"
        np.testing.assert_array_equal(terminal2.y, new_y)

    def test_kwargs_passed_to_model(self, regression_data):
        """Test kwargs are passed to model."""
        X_train, X_test, y, train_idx, test_idx = regression_data

        terminal = PredictTerminal(y=y, algorithm="ridge", kwargs={"alpha": 10.0})
        result = terminal.fit_evaluate(
            X_train, X_test, train_idx, test_idx, FittedStack()
        )

        # Just verify it runs without error
        assert isinstance(result, FoldResult)
