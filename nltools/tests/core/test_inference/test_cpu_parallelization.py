"""Tests for CPU parallelization functionality and correctness."""

import pytest
import numpy as np

from nltools.algorithms import (
    correlation_permutation_test,
    one_sample_permutation_test,
    two_sample_permutation_test,
)
from nltools.algorithms.inference import _timeseries_correlation_permutation_test


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
    return _timeseries_correlation_permutation_test(
        data1, data2, n_permute=60, return_null=True, n_jobs=n_jobs, random_state=42
    )


@pytest.mark.parametrize(
    ("run_at_n_jobs", "statistic_key"),
    [
        (_one_sample_at, "mean"),
    ],
    ids=["one_sample"],
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
