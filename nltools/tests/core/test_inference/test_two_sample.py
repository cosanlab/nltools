"""Tests for two-sample permutation tests."""

import pytest
import numpy as np

from nltools.algorithms import two_sample_permutation_test
from nltools.tests.core.test_inference import (
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.algorithms.backends import check_gpu_available


class TestTwoSamplePermutation:
    """Test two-sample permutation tests."""

    @pytest.mark.parametrize("n_features", [1, 10])
    def test_basic_functionality(self, n_features):
        """Test basic two-sample test with single or multiple features."""
        np.random.seed(42)
        if n_features == 1:
            data1 = np.random.randn(20)  # Group 1: 20 subjects
            data2 = np.random.randn(25)  # Group 2: 25 subjects
        else:
            data1 = np.random.randn(20, n_features)  # 20 subjects, n_features features
            data2 = np.random.randn(25, n_features)  # 25 subjects, n_features features

        result = two_sample_permutation_test(
            data1, data2, n_permute=100, random_state=42
        )

        assert "mean_diff" in result
        assert "p" in result
        assert "device" in result

        if n_features == 1:
            assert isinstance(result["mean_diff"], (float, np.floating))
            assert isinstance(result["p"], (float, np.floating))
            assert 0 <= result["p"] <= 1
        else:
            assert result["mean_diff"].shape == (n_features,)
            assert result["p"].shape == (n_features,)
            assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(25, 5)

        result1 = two_sample_permutation_test(
            data1, data2, n_permute=100, random_state=42
        )
        result2 = two_sample_permutation_test(
            data1, data2, n_permute=100, random_state=42
        )

        np.testing.assert_array_almost_equal(result1["mean_diff"], result2["mean_diff"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])

    @pytest.mark.parametrize("n_features", [1, 5])
    def test_return_null_distribution(self, n_features):
        """Test that null distribution is returned when requested."""
        np.random.seed(42)
        if n_features == 1:
            data1 = np.random.randn(20)
            data2 = np.random.randn(25)
            expected_shape = (100,)
        else:
            data1 = np.random.randn(20, n_features)
            data2 = np.random.randn(25, n_features)
            expected_shape = (100, n_features)

        result = two_sample_permutation_test(
            data1, data2, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == expected_shape

    def test_unequal_sample_sizes(self):
        """Test that unequal sample sizes work correctly."""
        np.random.seed(42)
        data1 = np.random.randn(15, 5)  # 15 subjects
        data2 = np.random.randn(35, 5)  # 35 subjects (different size)

        result = two_sample_permutation_test(
            data1, data2, n_permute=100, random_state=42
        )

        assert result["mean_diff"].shape == (5,)
        assert result["p"].shape == (5,)

    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        with pytest.raises(ValueError, match="tail must be"):
            two_sample_permutation_test(data1, data2, tail=3)

    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        data1 = np.random.randn(5, 5, 5)  # 3D
        data2 = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="data1 must be 1D to 2D"):
            two_sample_permutation_test(data1, data2)

    def test_mismatched_features(self):
        """Test that mismatched feature dimensions raise error."""
        data1 = np.random.randn(20, 5)  # 5 features
        data2 = np.random.randn(25, 10)  # 10 features (mismatch!)

        with pytest.raises(ValueError, match="must have same number of features"):
            two_sample_permutation_test(data1, data2)

    @pytest.mark.slow
    def test_cpu_parallel_correctness(self):
        """Test CPU parallelization produces correct results."""
        np.random.seed(42)
        data1 = np.random.randn(20, 50)
        data2 = np.random.randn(25, 50)

        result = two_sample_permutation_test(
            data1, data2, n_permute=500, device="cpu", n_jobs=2, random_state=42
        )

        # Mean difference should match observed
        obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)
        np.testing.assert_allclose(result["mean_diff"], obs_diff)

        # P-values should be valid
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert result["device"] == "cpu"

    @pytest.mark.slow
    @pytest.mark.gpu
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data1 = np.random.randn(20, 5000)
        data2 = np.random.randn(25, 5000)

        # NumPy backend
        result_numpy = two_sample_permutation_test(
            data1, data2, n_permute=500, device=None, random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = two_sample_permutation_test(
            data1,
            data2,
            n_permute=500,
            device="gpu",
            max_gpu_memory_gb=0.5,
            random_state=42,
        )

        # Results should match (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_numpy["mean_diff"],
            result_gpu["mean_diff"],
            rtol=TOLERANCE_GPU_VALUE,  # float32 vs float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=TOLERANCE_GPU_PVALUE,  # P-values accumulate more FP error
        )


# ============================================================================
# Test Two-Sample Permutation Statistical Correctness
# ============================================================================
