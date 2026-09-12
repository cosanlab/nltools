"""Tests for CPU parallelization functionality and correctness."""

import pytest
import numpy as np

from nltools.algorithms import (
    correlation_permutation_test,
    one_sample_permutation_test,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)


def _one_sample_at(n_jobs):
    data = np.random.RandomState(0).randn(20, 5)
    return one_sample_permutation_test(
        data, n_permute=60, return_null=True, n_jobs=n_jobs, random_state=42
    )


def _two_sample_at(n_jobs):
    rng = np.random.RandomState(0)
    data1, data2 = rng.randn(20, 5), rng.randn(18, 5)
    return two_sample_permutation_test(
        data1, data2, n_permute=60, return_null=True, n_jobs=n_jobs, random_state=42
    )


def _correlation_at(n_jobs):
    rng = np.random.RandomState(0)
    data1, data2 = rng.randn(25), rng.randn(25)
    return correlation_permutation_test(
        data1, data2, n_permute=60, return_null=True, n_jobs=n_jobs, random_state=42
    )


def _timeseries_at(n_jobs):
    rng = np.random.RandomState(0)
    data1, data2 = rng.randn(40), rng.randn(40)
    return timeseries_correlation_permutation_test(
        data1, data2, n_permute=60, return_null=True, n_jobs=n_jobs, random_state=42
    )


@pytest.mark.parametrize(
    ("run_at_n_jobs", "statistic_key"),
    [
        (_one_sample_at, "mean"),
        (_two_sample_at, "mean_diff"),
        (_correlation_at, "correlation"),
        (_timeseries_at, "correlation"),
    ],
    ids=["one_sample", "two_sample", "correlation", "timeseries"],
)
def test_worker_count_is_numerically_invisible(run_at_n_jobs, statistic_key):
    """Worker count never changes a seeded result: `n_jobs` is a speed knob only.

    Randomizations are drawn before the joblib block and consumed in index order,
    so one worker and every worker must agree bit for bit.
    """
    serial = run_at_n_jobs(1)
    parallel = run_at_n_jobs(-1)

    np.testing.assert_array_equal(serial["null_dist"], parallel["null_dist"])
    np.testing.assert_array_equal(serial[statistic_key], parallel[statistic_key])
    np.testing.assert_array_equal(serial["p"], parallel["p"])


@pytest.mark.slow
class TestCPUParallelization:
    """Test CPU parallelization functionality and correctness."""

    @pytest.mark.parametrize("n_features", [1, 100])
    def test_cpu_parallel(self, n_features):
        """Test CPU parallel with single or multiple features."""
        np.random.seed(42)
        if n_features == 1:
            data = np.random.randn(30)
        else:
            data = np.random.randn(30, n_features)

        result = one_sample_permutation_test(
            data, n_permute=500, n_jobs=2, random_state=42
        )

        # Verify results based on feature count
        if n_features == 1:
            assert isinstance(result["mean"], (float, np.floating))
            assert isinstance(result["p"], (float, np.floating))
        else:
            assert result["mean"].shape == (n_features,)
            assert result["p"].shape == (n_features,)
            assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    def test_cpu_parallel_correctness(self):
        """Test that CPU parallel produces statistically valid results."""
        np.random.seed(42)

        # Test with data that has mean = 0 (null hypothesis true)
        data_null = np.random.randn(30, 50)
        result_null = one_sample_permutation_test(
            data_null, n_permute=500, n_jobs=2, random_state=42
        )

        # Mean should match observed mean
        np.testing.assert_allclose(result_null["mean"], np.mean(data_null, axis=0))

        # P-values should be distributed (not all 0 or all 1)
        # Most should be non-significant (p > 0.05) since null is true
        assert np.sum(result_null["p"] > 0.05) > 40  # At least 80% non-significant

        # Test with data that has strong positive effect
        data_effect = np.random.randn(30, 50) + 2.0  # Mean = 2.0
        result_effect = one_sample_permutation_test(
            data_effect, n_permute=500, n_jobs=2, random_state=42
        )

        # All features should be significant (p < 0.05)
        assert np.all(result_effect["p"] < 0.05)

    def test_cpu_parallel_return_null(self):
        """Test that null distribution is returned correctly."""
        np.random.seed(42)
        data = np.random.randn(30, 10)

        result = one_sample_permutation_test(
            data,
            n_permute=200,
            n_jobs=2,
            return_null=True,
            random_state=42,
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (200, 10)

    def test_cpu_parallel_deterministic(self):
        """Test that CPU parallel is deterministic with same seed."""
        np.random.seed(42)
        data = np.random.randn(30, 10)

        result1 = one_sample_permutation_test(
            data, n_permute=200, n_jobs=2, random_state=42
        )
        result2 = one_sample_permutation_test(
            data, n_permute=200, n_jobs=2, random_state=42
        )

        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(result1["mean"], result2["mean"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])
