"""Tests for two-sample permutation tests."""

import pytest
import numpy as np

from nltools.algorithms import two_sample_permutation_test


class TestTwoSamplePermutation:
    """Test two-sample permutation tests."""

    @pytest.mark.parametrize("n_features", [10])
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

    @pytest.mark.parametrize("n_features", [5])
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

    def test_single_nan_matches_dropping_that_entry(self):
        """A NaN observation drops out of the mean difference and every permuted mean."""
        data1_with_nan = np.array([1.0, 2.0, 3.0, np.nan])
        data1_dropped = np.array([1.0, 2.0, 3.0])
        data2 = np.array([10.0, 11.0, 12.0, 13.0])

        result = two_sample_permutation_test(
            data1_with_nan, data2, n_permute=100, random_state=0
        )
        expected = two_sample_permutation_test(
            data1_dropped, data2, n_permute=100, random_state=0
        )

        assert np.isfinite(result["mean_diff"])
        assert result["mean_diff"] == pytest.approx(-9.5)
        assert result["mean_diff"] == pytest.approx(expected["mean_diff"])
        assert np.isfinite(result["p"])
        assert 0 < result["p"] <= 1


# ============================================================================
# Test Two-Sample Permutation Statistical Correctness
# ============================================================================
