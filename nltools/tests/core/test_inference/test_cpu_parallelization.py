"""Tests for CPU parallelization functionality and correctness."""

import pytest
import numpy as np

from nltools.algorithms import one_sample_permutation_test


def _one_sample_at(n_jobs):
    data = np.random.RandomState(0).randn(20, 5)
    return one_sample_permutation_test(
        data, n_permute=60, return_null=True, n_jobs=n_jobs, random_state=42
    )


def test_worker_count_is_numerically_invisible():
    """Worker count never changes a seeded result: `n_jobs` is a speed knob only.

    Randomizations are drawn before the joblib block and consumed in index order,
    so one worker and every worker must agree bit for bit.
    """
    serial = _one_sample_at(1)
    parallel = _one_sample_at(-1)

    np.testing.assert_array_equal(serial["null_dist"], parallel["null_dist"])
    np.testing.assert_array_equal(serial["mean"], parallel["mean"])
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
