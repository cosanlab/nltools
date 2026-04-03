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

pytestmark = pytest.mark.slow


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


class TestISCTerminal:
    """Tests for ISCTerminal."""

    def test_creation_defaults(self):
        """Test ISCTerminal creation with defaults."""
        from nltools.pipelines.terminals import ISCTerminal

        terminal = ISCTerminal()
        assert terminal.method == "pairwise"
        assert terminal.metric == "median"
        assert terminal.n_permute == 5000
        assert terminal.parallel == "cpu"

    def test_creation_custom(self):
        """Test ISCTerminal creation with custom parameters."""
        from nltools.pipelines.terminals import ISCTerminal

        terminal = ISCTerminal(
            method="leave-one-out",
            metric="mean",
            n_permute=1000,
            parallel=None,
        )
        assert terminal.method == "leave-one-out"
        assert terminal.metric == "mean"
        assert terminal.n_permute == 1000
        assert terminal.parallel is None

    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        from nltools.pipelines.terminals import ISCTerminal

        with pytest.raises(ValueError, match="method must be"):
            ISCTerminal(method="invalid")

    def test_invalid_metric_raises(self):
        """Test invalid metric raises ValueError."""
        from nltools.pipelines.terminals import ISCTerminal

        with pytest.raises(ValueError, match="metric must be"):
            ISCTerminal(metric="invalid")

    def test_invalid_parallel_raises(self):
        """Test invalid parallel raises ValueError."""
        from nltools.pipelines.terminals import ISCTerminal

        with pytest.raises(ValueError, match="parallel must be"):
            ISCTerminal(parallel="invalid")

    def test_fit_evaluate_single_feature(self):
        """Test ISC computation on single-feature data."""
        from nltools.pipelines.terminals import ISCTerminal
        from nltools.pipelines.results import ISCResult

        np.random.seed(42)

        # Create correlated data across subjects (shared signal + noise)
        n_obs, n_subjects = 50, 5
        shared_signal = np.random.randn(n_obs)
        data = [shared_signal + np.random.randn(n_obs) * 0.5 for _ in range(n_subjects)]

        terminal = ISCTerminal(method="pairwise", n_permute=100, parallel=None)
        result = terminal.fit_evaluate(data)

        assert isinstance(result, ISCResult)
        assert result.n_subjects == n_subjects
        assert result.method == "pairwise"
        assert result.metric == "median"
        # ISC should be positive for correlated data
        assert result.isc > 0

    def test_fit_evaluate_multi_feature(self):
        """Test ISC computation on multi-feature (voxel-wise) data."""
        from nltools.pipelines.terminals import ISCTerminal
        from nltools.pipelines.results import ISCResult

        np.random.seed(42)

        # Create data with multiple features (voxels)
        n_obs, n_subjects, n_features = 50, 5, 10
        shared_signal = np.random.randn(n_obs, n_features)
        data = [
            shared_signal + np.random.randn(n_obs, n_features) * 0.5
            for _ in range(n_subjects)
        ]

        terminal = ISCTerminal(method="leave-one-out", n_permute=100, parallel=None)
        result = terminal.fit_evaluate(data)

        assert isinstance(result, ISCResult)
        assert result.n_subjects == n_subjects
        assert result.method == "leave-one-out"
        # Should return per-voxel ISC
        assert result.isc.shape == (n_features,)
        assert result.p.shape == (n_features,)

    def test_fit_evaluate_insufficient_subjects_raises(self):
        """Test that less than 2 subjects raises error."""
        from nltools.pipelines.terminals import ISCTerminal

        terminal = ISCTerminal()
        with pytest.raises(ValueError, match="at least 2 subject"):
            terminal.fit_evaluate([np.random.randn(10)])

    def test_fit_evaluate_mismatched_observations_raises(self):
        """Test that mismatched observation counts raise error."""
        from nltools.pipelines.terminals import ISCTerminal

        terminal = ISCTerminal()
        data = [np.random.randn(50), np.random.randn(60)]  # Different n_obs
        with pytest.raises(ValueError, match="same number of observations"):
            terminal.fit_evaluate(data)


class TestRSATerminal:
    """Tests for RSATerminal."""

    def test_creation_defaults(self):
        """Test RSATerminal creation with defaults."""
        from nltools.pipelines.terminals import RSATerminal

        model_rdm = np.random.rand(5, 5)
        model_rdm = (model_rdm + model_rdm.T) / 2  # Make symmetric
        terminal = RSATerminal(model_rdm=model_rdm)

        assert terminal.method == "spearman"
        assert terminal.n_permute == 5000

    def test_creation_custom(self):
        """Test RSATerminal creation with custom parameters."""
        from nltools.pipelines.terminals import RSATerminal

        model_rdm = np.random.rand(5, 5)
        model_rdm = (model_rdm + model_rdm.T) / 2
        terminal = RSATerminal(
            model_rdm=model_rdm,
            method="pearson",
            n_permute=1000,
        )
        assert terminal.method == "pearson"
        assert terminal.n_permute == 1000

    def test_condensed_model_rdm(self):
        """Test RSATerminal accepts condensed RDM."""
        from nltools.pipelines.terminals import RSATerminal
        from scipy.spatial.distance import squareform

        n_conditions = 5
        model_square = np.random.rand(n_conditions, n_conditions)
        model_square = (model_square + model_square.T) / 2
        model_condensed = squareform(model_square, checks=False)

        terminal = RSATerminal(model_rdm=model_condensed)
        # Should be stored in condensed form
        assert terminal.model_rdm.ndim == 1
        assert len(terminal.model_rdm) == n_conditions * (n_conditions - 1) // 2

    def test_invalid_method_raises(self):
        """Test invalid method raises ValueError."""
        from nltools.pipelines.terminals import RSATerminal

        with pytest.raises(ValueError, match="method must be"):
            RSATerminal(model_rdm=np.eye(5), method="invalid")

    def test_non_square_2d_raises(self):
        """Test non-square 2D model RDM raises ValueError."""
        from nltools.pipelines.terminals import RSATerminal

        with pytest.raises(ValueError, match="must be square"):
            RSATerminal(model_rdm=np.random.rand(5, 3))

    def test_fit_evaluate_square_rdm(self):
        """Test RSA computation with square neural RDM."""
        from nltools.pipelines.terminals import RSATerminal
        from nltools.pipelines.results import RSAResult

        np.random.seed(42)

        # Create model and neural RDMs with some correlation
        n_conditions = 8
        base = np.random.rand(n_conditions, n_conditions)
        model_rdm = (base + base.T) / 2

        # Neural RDM = model + noise (should correlate)
        noise = np.random.rand(n_conditions, n_conditions) * 0.3
        noise = (noise + noise.T) / 2
        neural_rdm = model_rdm + noise

        terminal = RSATerminal(model_rdm=model_rdm, n_permute=100)
        result = terminal.fit_evaluate(neural_rdm)

        assert isinstance(result, RSAResult)
        assert result.method == "spearman"
        assert result.n_conditions == n_conditions
        # Should have positive correlation for similar RDMs
        assert result.correlation > 0

    def test_fit_evaluate_condensed_rdm(self):
        """Test RSA computation with condensed neural RDM."""
        from nltools.pipelines.terminals import RSATerminal
        from nltools.pipelines.results import RSAResult
        from scipy.spatial.distance import squareform

        np.random.seed(42)

        n_conditions = 6
        base = np.random.rand(n_conditions, n_conditions)
        model_rdm = (base + base.T) / 2
        neural_rdm = model_rdm + np.random.rand(n_conditions, n_conditions) * 0.2
        neural_rdm = (neural_rdm + neural_rdm.T) / 2
        neural_condensed = squareform(neural_rdm, checks=False)

        terminal = RSATerminal(model_rdm=model_rdm, n_permute=100)
        result = terminal.fit_evaluate(neural_condensed)

        assert isinstance(result, RSAResult)
        assert result.n_conditions == n_conditions

    def test_fit_evaluate_from_features(self):
        """Test RSA computation from condition x features matrix."""
        from nltools.pipelines.terminals import RSATerminal
        from nltools.pipelines.results import RSAResult

        np.random.seed(42)

        n_conditions, n_features = 5, 50

        # Create features that produce predictable RDM
        features = np.random.randn(n_conditions, n_features)

        # Create model RDM that matches the correlation distance structure
        from sklearn.metrics import pairwise_distances

        model_dist = pairwise_distances(features, metric="correlation")
        # Make symmetric (should already be, but just in case)
        model_rdm = (model_dist + model_dist.T) / 2

        terminal = RSATerminal(model_rdm=model_rdm, n_permute=100)
        result = terminal.fit_evaluate(features)

        assert isinstance(result, RSAResult)
        assert result.n_conditions == n_conditions
        # Should have perfect correlation since RDM computed from same features
        assert result.correlation > 0.99

    def test_fit_evaluate_size_mismatch_raises(self):
        """Test that RDM size mismatch raises error."""
        from nltools.pipelines.terminals import RSATerminal

        model_rdm = np.eye(5)  # 5 conditions
        neural_rdm = np.eye(6)  # 6 conditions

        terminal = RSATerminal(model_rdm=model_rdm, n_permute=100)
        with pytest.raises(ValueError, match="RDM size mismatch"):
            terminal.fit_evaluate(neural_rdm)

    def test_fit_evaluate_pearson(self):
        """Test RSA with Pearson correlation."""
        from nltools.pipelines.terminals import RSATerminal
        from nltools.pipelines.results import RSAResult

        np.random.seed(42)

        n_conditions = 5
        model_rdm = np.eye(n_conditions)
        neural_rdm = (
            np.eye(n_conditions) + np.random.rand(n_conditions, n_conditions) * 0.1
        )
        neural_rdm = (neural_rdm + neural_rdm.T) / 2

        terminal = RSATerminal(model_rdm=model_rdm, method="pearson", n_permute=100)
        result = terminal.fit_evaluate(neural_rdm)

        assert isinstance(result, RSAResult)
        assert result.method == "pearson"

    def test_fit_evaluate_kendall(self):
        """Test RSA with Kendall correlation."""
        from nltools.pipelines.terminals import RSATerminal
        from nltools.pipelines.results import RSAResult

        np.random.seed(42)

        n_conditions = 5
        model_rdm = np.eye(n_conditions)
        neural_rdm = (
            np.eye(n_conditions) + np.random.rand(n_conditions, n_conditions) * 0.1
        )
        neural_rdm = (neural_rdm + neural_rdm.T) / 2

        terminal = RSATerminal(model_rdm=model_rdm, method="kendall", n_permute=100)
        result = terminal.fit_evaluate(neural_rdm)

        assert isinstance(result, RSAResult)
        assert result.method == "kendall"


class TestISCResult:
    """Tests for ISCResult dataclass."""

    def test_repr_scalar(self):
        """Test repr for scalar ISC."""
        from nltools.pipelines.results import ISCResult

        result = ISCResult(
            isc=np.float64(0.5),
            p=np.float64(0.01),
            ci=(0.4, 0.6),
            method="pairwise",
            metric="median",
            n_subjects=10,
        )
        r = repr(result)
        assert "0.5" in r
        assert "0.01" in r

    def test_repr_array(self):
        """Test repr for array ISC (voxel-wise)."""
        from nltools.pipelines.results import ISCResult

        isc = np.array([0.3, 0.5, 0.7])
        p = np.array([0.1, 0.01, 0.001])
        result = ISCResult(
            isc=isc,
            p=p,
            ci=(np.array([0.2, 0.4, 0.6]), np.array([0.4, 0.6, 0.8])),
            method="leave-one-out",
            metric="mean",
            n_subjects=5,
        )
        r = repr(result)
        assert "n_voxels=3" in r
        assert "sig_voxels=2" in r  # p < 0.05 for 2 voxels

    def test_summary_scalar(self):
        """Test summary for scalar ISC."""
        from nltools.pipelines.results import ISCResult

        result = ISCResult(
            isc=np.float64(0.5),
            p=np.float64(0.01),
            ci=(0.4, 0.6),
            method="pairwise",
            metric="median",
            n_subjects=10,
        )
        summary = result.summary()
        assert "pairwise" in summary
        assert "median" in summary
        assert "10" in summary


class TestRSAResult:
    """Tests for RSAResult dataclass."""

    def test_repr(self):
        """Test repr for RSA result."""
        from nltools.pipelines.results import RSAResult

        result = RSAResult(
            correlation=0.65,
            p_value=0.001,
            ci=(0.5, 0.8),
            method="spearman",
            n_conditions=8,
        )
        r = repr(result)
        assert "0.65" in r
        assert "0.001" in r

    def test_summary(self):
        """Test summary for RSA result."""
        from nltools.pipelines.results import RSAResult

        result = RSAResult(
            correlation=0.65,
            p_value=0.001,
            ci=(0.5, 0.8),
            method="spearman",
            n_conditions=8,
        )
        summary = result.summary()
        assert "spearman" in summary
        assert "8" in summary
        assert "0.65" in summary
