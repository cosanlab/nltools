"""
Tests for nltools.stats.permutation — user-facing permutation test API.

These tests verify that stats.permutation facades correctly delegate to
algorithms.inference and that users can import from nltools.stats.
"""

import importlib

import numpy as np
import pytest


PERMUTATION_EXPORTS = [
    "one_sample_permutation_test",
    "two_sample_permutation_test",
    "correlation_permutation_test",
    "timeseries_correlation_permutation_test",
    "circle_shift",
    "phase_randomize",
    "matrix_permutation_test",
    "double_center",
    "u_center",
    "distance_correlation",
]


class TestStatsPermutationImports:
    """Verify all permutation functions are importable from nltools.stats."""

    @pytest.mark.parametrize("name", PERMUTATION_EXPORTS)
    @pytest.mark.parametrize("module", ["nltools.stats", "nltools.stats.permutation"])
    def test_export_is_importable(self, module, name):
        """Each permutation function is exposed by both public modules."""
        assert hasattr(importlib.import_module(module), name)


class TestOneSamplePermutation:
    """Test one_sample_permutation_test facade produces correct results."""

    def test_basic(self):
        from nltools.stats import one_sample_permutation_test

        rng = np.random.RandomState(42)
        data = rng.randn(30)
        result = one_sample_permutation_test(data, n_permute=100, random_state=42)

        assert "mean" in result
        assert "p" in result
        assert isinstance(result["p"], float)

    def test_matches_algorithms(self):
        """Stats facade must return identical results to algorithms.inference."""
        from nltools.stats import one_sample_permutation_test as stats_fn
        from nltools.algorithms.inference import one_sample_permutation_test as algo_fn

        rng = np.random.RandomState(42)
        data = rng.randn(30, 5)

        stats_result = stats_fn(data, n_permute=100, random_state=42)
        algo_result = algo_fn(data, n_permute=100, random_state=42)

        np.testing.assert_array_equal(stats_result["mean"], algo_result["mean"])
        np.testing.assert_array_equal(stats_result["p"], algo_result["p"])

    def test_return_null(self):
        from nltools.stats import one_sample_permutation_test

        data = np.random.RandomState(42).randn(20)
        result = one_sample_permutation_test(
            data, n_permute=50, return_null=True, random_state=42
        )
        assert "null_dist" in result
        assert result["null_dist"].shape == (50,)


class TestTwoSamplePermutation:
    """Test two_sample_permutation_test facade."""

    def test_basic(self):
        from nltools.stats import two_sample_permutation_test

        rng = np.random.RandomState(42)
        data1 = rng.randn(20)
        data2 = rng.randn(20) + 1
        result = two_sample_permutation_test(
            data1, data2, n_permute=100, random_state=42
        )

        assert "mean_diff" in result
        assert "p" in result

    def test_matches_algorithms(self):
        from nltools.stats import two_sample_permutation_test as stats_fn
        from nltools.algorithms.inference import two_sample_permutation_test as algo_fn

        rng = np.random.RandomState(42)
        data1 = rng.randn(15, 3)
        data2 = rng.randn(15, 3)

        stats_result = stats_fn(data1, data2, n_permute=100, random_state=42)
        algo_result = algo_fn(data1, data2, n_permute=100, random_state=42)

        np.testing.assert_array_equal(
            stats_result["mean_diff"], algo_result["mean_diff"]
        )
        np.testing.assert_array_equal(stats_result["p"], algo_result["p"])


class TestCorrelationPermutation:
    """Test correlation_permutation_test facade."""

    def test_basic(self):
        from nltools.stats import correlation_permutation_test

        rng = np.random.RandomState(42)
        x = rng.randn(30)
        y = rng.randn(30)
        result = correlation_permutation_test(x, y, n_permute=100, random_state=42)

        assert "correlation" in result
        assert "p" in result

    def test_matches_algorithms(self):
        from nltools.stats import correlation_permutation_test as stats_fn
        from nltools.algorithms.inference import correlation_permutation_test as algo_fn

        rng = np.random.RandomState(42)
        x = rng.randn(30)
        y = rng.randn(30)

        stats_result = stats_fn(x, y, n_permute=100, random_state=42)
        algo_result = algo_fn(x, y, n_permute=100, random_state=42)

        np.testing.assert_equal(stats_result["correlation"], algo_result["correlation"])
        np.testing.assert_equal(stats_result["p"], algo_result["p"])


class TestTimeseriesCorrelationPermutation:
    """Test timeseries_correlation_permutation_test facade."""

    def test_basic_circle_shift(self):
        from nltools.stats import timeseries_correlation_permutation_test

        rng = np.random.RandomState(42)
        x = rng.randn(100)
        y = rng.randn(100)
        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=50, random_state=42
        )

        assert "correlation" in result
        assert "p" in result


class TestCircleShift:
    """Test circle_shift facade."""

    def test_basic(self):
        from nltools.stats import circle_shift

        data = np.arange(10, dtype=float)
        shifted = circle_shift(data, shift_amount=3)
        assert shifted.shape == data.shape
        np.testing.assert_array_equal(shifted, np.roll(data, 3))


class TestPhaseRandomize:
    """Test phase_randomize facade."""

    def test_basic(self):
        from nltools.stats import phase_randomize

        rng = np.random.RandomState(42)
        data = rng.randn(100)
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape
        # Power spectrum should be preserved
        np.testing.assert_allclose(
            np.abs(np.fft.fft(data)),
            np.abs(np.fft.fft(randomized)),
            atol=1e-10,
        )


class TestMatrixPermutation:
    """Test matrix_permutation_test facade."""

    def test_basic(self):
        from nltools.stats import matrix_permutation_test

        rng = np.random.RandomState(42)
        n = 10
        mat1 = rng.randn(n, n)
        mat1 = (mat1 + mat1.T) / 2
        mat2 = rng.randn(n, n)
        mat2 = (mat2 + mat2.T) / 2
        np.fill_diagonal(mat1, 0)
        np.fill_diagonal(mat2, 0)

        result = matrix_permutation_test(mat1, mat2, n_permute=100, random_state=42)

        assert "correlation" in result
        assert "p" in result


class TestDoubleCenter:
    """Test double_center facade."""

    def test_basic(self):
        from nltools.stats import double_center

        mat = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        centered = double_center(mat)
        assert centered.shape == mat.shape
        # Row and column means should be zero
        np.testing.assert_allclose(centered.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(centered.mean(axis=1), 0, atol=1e-10)


class TestUCenter:
    """Test u_center facade."""

    def test_basic(self):
        from nltools.stats import u_center

        mat = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
        centered = u_center(mat)
        assert centered.shape == mat.shape


class TestDistanceCorrelation:
    """Test distance_correlation facade."""

    def test_basic(self):
        from nltools.stats import distance_correlation

        rng = np.random.RandomState(42)
        x = rng.randn(20)
        y = x + rng.randn(20) * 0.1  # Strongly correlated
        result = distance_correlation(x, y)

        assert "dcorr" in result
        assert result["dcorr"] > 0.5  # Should detect strong correlation
