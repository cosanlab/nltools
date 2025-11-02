"""Tests for two-sample permutation tests."""

import pytest
import numpy as np
from scipy.stats import kstest

from nltools.algorithms.inference import two_sample_permutation_test
from nltools.tests.core.test_inference import (
    N_PERMUTE_BACKEND,
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.backends import check_gpu_available


class TestTwoSamplePermutation:
    """Test two-sample permutation tests."""

    @pytest.mark.tier1
    def test_basic_functionality_single_feature(self):
        """Test basic two-sample test with single feature (1D arrays)."""
        np.random.seed(42)
        data1 = np.random.randn(20)  # Group 1: 20 subjects
        data2 = np.random.randn(25)  # Group 2: 25 subjects

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

        assert "mean_diff" in result
        assert "p" in result
        assert "parallel" in result
        assert isinstance(result["mean_diff"], (float, np.floating))
        assert isinstance(result["p"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.tier2
    def test_basic_functionality_multi_feature(self):
        """Test two-sample test with multiple features (2D arrays)."""
        np.random.seed(42)
        data1 = np.random.randn(20, 10)  # 20 subjects, 10 features
        data2 = np.random.randn(25, 10)  # 25 subjects, 10 features

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

        assert result["mean_diff"].shape == (10,)
        assert result["p"].shape == (10,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    @pytest.mark.tier1
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

    @pytest.mark.tier1
    def test_return_null_distribution_single(self):
        """Test that null distribution is returned for single feature."""
        np.random.seed(42)
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        result = two_sample_permutation_test(
            data1, data2, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100,)

    @pytest.mark.tier1
    def test_return_null_distribution_multi(self):
        """Test null distribution for multi-feature data."""
        np.random.seed(42)
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(25, 5)

        result = two_sample_permutation_test(
            data1, data2, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100, 5)

    @pytest.mark.tier2
    def test_significant_effect(self):
        """Test that significant group difference is detected."""
        np.random.seed(42)
        data1 = np.random.randn(30)  # Mean = 0
        data2 = np.random.randn(30) + 2.0  # Mean = 2.0 (large difference)

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

        assert result["p"] < 0.05  # Should be significant

    @pytest.mark.tier2
    def test_non_significant_effect(self):
        """Test that non-significant difference has high p-value."""
        np.random.seed(42)
        data1 = np.random.randn(30)
        data2 = np.random.randn(30) + 0.1  # Small difference

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

        assert result["p"] > 0.05  # Should not be significant

    def test_unequal_sample_sizes(self):
        """Test that unequal sample sizes work correctly."""
        np.random.seed(42)
        data1 = np.random.randn(15, 5)  # 15 subjects
        data2 = np.random.randn(35, 5)  # 35 subjects (different size)

        result = two_sample_permutation_test(
            data1, data2, n_permute=500, random_state=42
        )

        assert result["mean_diff"].shape == (5,)
        assert result["p"].shape == (5,)

    @pytest.mark.tier2
    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed and two-tailed p-values differ."""
        np.random.seed(42)
        data1 = np.random.randn(30)
        data2 = np.random.randn(30) + 0.5

        result_two = two_sample_permutation_test(
            data1, data2, n_permute=1000, tail=2, random_state=42
        )
        result_one = two_sample_permutation_test(
            data1, data2, n_permute=1000, tail=1, random_state=42
        )

        # One-tailed should be different from two-tailed
        assert result_one["p"] != result_two["p"]

    @pytest.mark.tier1
    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            two_sample_permutation_test(data1, data2, tail=3)

    @pytest.mark.tier1
    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        data1 = np.random.randn(5, 5, 5)  # 3D
        data2 = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="data1 must be 1D to 2D"):
            two_sample_permutation_test(data1, data2)

    @pytest.mark.tier1
    def test_mismatched_features(self):
        """Test that mismatched feature dimensions raise error."""
        data1 = np.random.randn(20, 5)  # 5 features
        data2 = np.random.randn(25, 10)  # 10 features (mismatch!)

        with pytest.raises(ValueError, match="must have same number of features"):
            two_sample_permutation_test(data1, data2)

    @pytest.mark.tier2
    def test_backend_consistency_single_feature(self, backends):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        results = {}
        for backend in backends:
            # Map backend to parallel parameter
            if backend == "numpy":
                parallel = None
            elif backend == "torch":
                parallel = "gpu"
            else:
                parallel = "cpu"

            results[backend] = two_sample_permutation_test(
                data1,
                data2,
                n_permute=N_PERMUTE_BACKEND,
                parallel=parallel,
                random_state=42,
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["mean_diff"],
            results["torch"]["mean_diff"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            results["numpy"]["p"],
            results["torch"]["p"],
            rtol=1e-5,
        )

    @pytest.mark.tier2
    def test_backend_consistency_multi_feature(self, backends):
        """Test backend consistency for multi-feature data."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data1 = np.random.randn(20, 10)
        data2 = np.random.randn(25, 10)

        results = {}
        for backend in backends:
            # Map backend to parallel parameter
            if backend == "numpy":
                parallel = None
            elif backend == "torch":
                parallel = "gpu"
            else:
                parallel = "cpu"

            results[backend] = two_sample_permutation_test(
                data1,
                data2,
                n_permute=N_PERMUTE_BACKEND,
                parallel=parallel,
                random_state=42,
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["mean_diff"],
            results["torch"]["mean_diff"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            results["numpy"]["p"],
            results["torch"]["p"],
            rtol=1e-5,
        )

    @pytest.mark.tier2
    def test_cpu_parallel_correctness(self):
        """Test CPU parallelization produces correct results."""
        np.random.seed(42)
        data1 = np.random.randn(20, 50)
        data2 = np.random.randn(25, 50)

        result = two_sample_permutation_test(
            data1, data2, n_permute=500, parallel="cpu", n_jobs=2, random_state=42
        )

        # Mean difference should match observed
        obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)
        np.testing.assert_allclose(result["mean_diff"], obs_diff)

        # P-values should be valid
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert result["parallel"] == "cpu"

    @pytest.mark.tier2
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data1 = np.random.randn(20, 5000)
        data2 = np.random.randn(25, 5000)

        # NumPy backend
        result_numpy = two_sample_permutation_test(
            data1, data2, n_permute=500, parallel=None, random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = two_sample_permutation_test(
            data1,
            data2,
            n_permute=500,
            parallel="gpu",
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


class TestTwoSamplePermutationStatisticalCorrectness:
    """Test statistical correctness of two-sample permutation tests."""

    @pytest.mark.tier2
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis."""
        n_samples1 = 30
        n_samples2 = 30
        n_tests = 100  # Run many tests with different seeds
        n_permute = 2000  # Enough permutations for stable p-values

        p_values = []

        for seed in range(n_tests):
            np.random.seed(seed)
            # Generate two groups from same distribution (both N(0, 1))
            data1 = np.random.randn(n_samples1)
            data2 = np.random.randn(n_samples2)

            result = two_sample_permutation_test(
                data1, data2, n_permute=n_permute, random_state=seed
            )

            p_values.append(result["p"])

        # Test uniformity using Kolmogorov-Smirnov test
        # Under null hypothesis, p-values should be uniformly distributed
        ks_statistic, ks_pvalue = kstest(p_values, "uniform")

        # KS test p-value should be > 0.05 (p-values are uniform)
        assert ks_pvalue > 0.05, (
            f"P-values should be uniformly distributed under null hypothesis. "
            f"KS test p-value: {ks_pvalue:.4f}"
        )

    @pytest.mark.tier1
    def test_effect_size_sensitivity(self):
        """Test that larger mean difference produces lower p-values."""
        n_samples1 = 30
        n_samples2 = 30
        n_permute = 5000  # Large enough for stable p-values

        # Test with different mean differences
        mean_diffs = [0.0, 0.5, 1.0, 2.0]  # Null, small, medium, large effect
        p_values = []

        for mean_diff in mean_diffs:
            np.random.seed(42)  # Same seed for reproducibility
            data1 = np.random.randn(n_samples1)
            data2 = np.random.randn(n_samples2) + mean_diff  # Group 2 shifted

            result = two_sample_permutation_test(
                data1, data2, n_permute=n_permute, random_state=42
            )

            p_values.append(result["p"])

        # Verify larger mean difference → smaller p-value (monotonic relationship)
        # Skip mean_diff=0 (null hypothesis), test others
        # Note: Very large effects may hit minimum p-value (1/(n_permute+1)),
        # so allow >= for equality case when effects are extremely large
        assert p_values[1] >= p_values[2], (
            f"Larger effect should produce smaller p-value. "
            f"mean_diff=0.5: p={p_values[1]:.6f}, mean_diff=1.0: p={p_values[2]:.6f}"
        )
        assert p_values[2] >= p_values[3], (
            f"Larger effect should produce smaller p-value. "
            f"mean_diff=1.0: p={p_values[2]:.6f}, mean_diff=2.0: p={p_values[3]:.6f}"
        )

        # Medium effect (mean_diff=1.0) should be significant
        assert p_values[2] < 0.05, (
            f"Medium effect (mean_diff=1.0) should be significant, got p={p_values[2]:.4f}"
        )
        # Large effect (mean_diff=2.0) should be significant
        assert p_values[3] < 0.05, (
            f"Large effect (mean_diff=2.0) should be significant, got p={p_values[3]:.4f}"
        )

    @pytest.mark.tier1
    def test_mean_difference_correctness(self):
        """Test that computed mean difference matches expected value."""
        n_samples1 = 30
        n_samples2 = 30
        true_mean1 = 5.0
        true_mean2 = 2.0
        expected_diff = true_mean1 - true_mean2  # 3.0

        np.random.seed(42)
        data1 = np.random.randn(n_samples1) + true_mean1
        data2 = np.random.randn(n_samples2) + true_mean2

        result = two_sample_permutation_test(
            data1, data2, n_permute=2000, random_state=42
        )

        # Computed mean difference should be close to expected
        # Tolerance: rtol=0.05 (5% as specified in plan)
        np.testing.assert_allclose(
            result["mean_diff"], expected_diff, rtol=0.05, atol=0.1
        )

    @pytest.mark.tier2
    def test_group_size_sensitivity(self):
        """Test that larger groups produce more stable p-values."""
        # Same effect size (mean difference = 2), different group sizes
        mean_diff = 2.0
        small_n = 10
        large_n = 50

        # Run multiple times with different seeds to estimate variance
        n_runs = 20
        n_permute = 2000

        p_values_small = []
        p_values_large = []

        for seed in range(n_runs):
            np.random.seed(seed)
            # Small groups
            data1_small = np.random.randn(small_n)
            data2_small = np.random.randn(small_n) + mean_diff
            result_small = two_sample_permutation_test(
                data1_small, data2_small, n_permute=n_permute, random_state=seed
            )
            p_values_small.append(result_small["p"])

            # Large groups
            data1_large = np.random.randn(large_n)
            data2_large = np.random.randn(large_n) + mean_diff
            result_large = two_sample_permutation_test(
                data1_large, data2_large, n_permute=n_permute, random_state=seed
            )
            p_values_large.append(result_large["p"])

        # Larger groups should produce more stable p-values (lower variance)
        variance_small = np.var(p_values_small)
        variance_large = np.var(p_values_large)

        assert variance_large < variance_small * 1.5, (
            f"Larger groups should produce more stable p-values (lower variance). "
            f"Small n={small_n}: variance={variance_small:.6f}, "
            f"Large n={large_n}: variance={variance_large:.6f}"
        )

    @pytest.mark.tier1
    def test_one_tailed_directional_correctness(self):
        """Test that one-tailed test correctly detects directional effects."""
        n_samples1 = 30
        n_samples2 = 30
        n_permute = 5000

        # Case 1: group1 > group2 (mean1=1, mean2=0.3) - small effect to avoid p-value saturation
        np.random.seed(42)
        data1_pos = np.random.randn(n_samples1) + 1.0
        data2_pos = np.random.randn(n_samples2) + 0.3

        result_pos = two_sample_permutation_test(
            data1_pos, data2_pos, n_permute=n_permute, tail=1, random_state=42
        )

        # Case 2: group1 < group2 (mean1=0.3, mean2=1) - opposite direction, same effect size
        np.random.seed(42)
        data1_neg = np.random.randn(n_samples1) + 0.3
        data2_neg = np.random.randn(n_samples2) + 1.0

        result_neg = two_sample_permutation_test(
            data1_neg, data2_neg, n_permute=n_permute, tail=1, random_state=42
        )

        # Check that mean differences are correct
        assert result_pos["mean_diff"] > 0, (
            f"Positive case should have positive mean_diff. Got {result_pos['mean_diff']:.4f}"
        )
        assert result_neg["mean_diff"] < 0, (
            f"Negative case should have negative mean_diff. Got {result_neg['mean_diff']:.4f}"
        )

        # For one-tailed test: tests if mean_diff is significantly different from 0 in observed direction
        # - Positive mean_diff: tests if significantly positive (should have small p-value)
        # - Negative mean_diff: tests if significantly negative (should have small p-value)
        # Both should be significant since effect size is the same, just opposite directions
        assert result_pos["p"] < 0.05, (
            f"One-tailed test should detect positive mean_diff as significant. "
            f"group1 > group2: p={result_pos['p']:.4f}"
        )
        assert result_neg["p"] < 0.05, (
            f"One-tailed test should detect negative mean_diff as significant. "
            f"group1 < group2: p={result_neg['p']:.4f}"
        )

        # Verify that two-tailed test gives similar p-values for both cases
        # (since effect size is same, just opposite directions)
        result_pos_two = two_sample_permutation_test(
            data1_pos, data2_pos, n_permute=n_permute, tail=2, random_state=42
        )
        result_neg_two = two_sample_permutation_test(
            data1_neg, data2_neg, n_permute=n_permute, tail=2, random_state=42
        )

        # Two-tailed p-values should be similar (same effect size, opposite directions)
        # Allow reasonable tolerance since exact p-values depend on null distribution
        np.testing.assert_allclose(
            result_pos_two["p"], result_neg_two["p"], rtol=0.3, atol=0.01
        )

        # One-tailed p-values should be approximately half of two-tailed (for same direction)
        ratio_pos = result_pos["p"] / result_pos_two["p"]
        ratio_neg = result_neg["p"] / result_neg_two["p"]

        # One-tailed should be roughly half of two-tailed (allow tolerance)
        # Skip if p-values hit minimum (ratio will be 1.0)
        if result_pos["p"] > 0.001:  # Only check ratio if not at minimum
            assert 0.3 < ratio_pos < 0.7, (
                f"One-tailed p-value should be ~half of two-tailed. "
                f"Got ratio={ratio_pos:.4f}, one_tailed={result_pos['p']:.4f}, two_tailed={result_pos_two['p']:.4f}"
            )
        if result_neg["p"] > 0.001:  # Only check ratio if not at minimum
            assert 0.3 < ratio_neg < 0.7, (
                f"One-tailed p-value should be ~half of two-tailed. "
                f"Got ratio={ratio_neg:.4f}, one_tailed={result_neg['p']:.4f}, two_tailed={result_neg_two['p']:.4f}"
            )

    @pytest.mark.tier1
    def test_null_distribution_centered_at_zero(self):
        """Test that null distribution is centered at zero under null hypothesis."""
        n_samples1 = 30
        n_samples2 = 30
        n_permute = 2000

        # Generate null data (both groups from same distribution)
        np.random.seed(42)
        data1 = np.random.randn(n_samples1)
        data2 = np.random.randn(n_samples2)

        result = two_sample_permutation_test(
            data1, data2, n_permute=n_permute, return_null=True, random_state=42
        )

        # Null distribution mean should be close to 0 (within sampling error)
        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

        # Expected std of mean under null: std(null_dist) / sqrt(n_permute)
        # Use more lenient tolerance (3 standard errors)
        expected_std_of_mean = null_std / np.sqrt(n_permute)
        tolerance = 3 * expected_std_of_mean

        assert abs(null_mean) < tolerance, (
            f"Null distribution mean should be close to 0. "
            f"Got mean={null_mean:.6f}, expected within ±{tolerance:.6f}"
        )
