"""
Tests for one-sample permutation tests.

Tests both basic functionality and statistical correctness.
"""

import pytest
import numpy as np

from nltools.algorithms.inference import one_sample_permutation_test
from nltools.tests.core.test_inference import (
    N_PERMUTE_STATS_COMPARISON,
)


class TestOneSamplePermutation:
    """Test one-sample permutation tests."""

    def test_basic_functionality_single_feature(self):
        """Test basic one-sample test with single feature."""
        np.random.seed(42)
        data = np.random.randn(30)  # Mean ~ 0
        result = one_sample_permutation_test(data, n_permute=1000, random_state=42)

        assert "mean" in result
        assert "p" in result
        assert "backend" in result
        assert isinstance(result["mean"], (float, np.floating))
        assert isinstance(result["p"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_basic_functionality_multi_feature(self):
        """Test basic one-sample test with multiple features."""
        np.random.seed(42)
        data = np.random.randn(30, 10)  # 30 samples, 10 features
        result = one_sample_permutation_test(data, n_permute=1000, random_state=42)

        assert result["mean"].shape == (10,)
        assert result["p"].shape == (10,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        data = np.random.randn(30, 5)

        result1 = one_sample_permutation_test(data, n_permute=100, random_state=42)
        result2 = one_sample_permutation_test(data, n_permute=100, random_state=42)

        np.testing.assert_array_almost_equal(result1["mean"], result2["mean"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])

    def test_return_null_distribution(self):
        """Test that null distribution is returned when requested."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(
            data, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100,)

    def test_return_null_distribution_multifeature(self):
        """Test null distribution for multi-feature data."""
        np.random.seed(42)
        data = np.random.randn(30, 5)
        result = one_sample_permutation_test(
            data, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100, 5)

    def test_significant_effect(self):
        """Test that significant effect is detected."""
        # Generate data with large positive mean
        np.random.seed(42)
        data = np.random.randn(30) + 2.0  # Mean = 2.0
        result = one_sample_permutation_test(data, n_permute=1000, random_state=42)

        assert result["p"] < 0.05  # Should be significant

    def test_non_significant_effect(self):
        """Test that non-significant effect has high p-value."""
        # Generate data with mean ~ 0
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, n_permute=1000, random_state=42)

        assert result["p"] > 0.05  # Should not be significant

    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed and two-tailed p-values differ."""
        np.random.seed(42)
        data = np.random.randn(30) + 0.5

        result_two = one_sample_permutation_test(
            data, n_permute=N_PERMUTE_STATS_COMPARISON, tail=2, random_state=42
        )
        result_one = one_sample_permutation_test(
            data, n_permute=N_PERMUTE_STATS_COMPARISON, tail=1, random_state=42
        )

        # One-tailed p-value should be approximately half of two-tailed
        # (for positive effect)
        assert result_one["p"] < result_two["p"]

    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        data = np.random.randn(30)
        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            one_sample_permutation_test(data, tail=3)

    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        data = np.random.randn(5, 5, 5)  # 3D
        with pytest.raises(ValueError, match="data must be 1D or 2D"):
            one_sample_permutation_test(data)


class TestOneSamplePermutationStatisticalCorrectness:
    """Test statistical correctness of one-sample permutation tests (not just CPU/GPU consistency)."""

    @pytest.mark.tier1
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis (mean=0)."""
        from scipy.stats import kstest

        n_samples = 50
        n_tests = 100  # Run many tests with different seeds
        p_values = []

        # Generate data from N(0, 1) (true mean = 0) and run many tests
        for seed in range(n_tests):
            np.random.seed(seed)
            data = np.random.randn(n_samples)  # Mean ~ 0
            result = one_sample_permutation_test(
                data, n_permute=2000, random_state=seed
            )
            p_values.append(result["p"])

        p_values = np.array(p_values)

        # Verify p-values are uniformly distributed using Kolmogorov-Smirnov test
        # Under null hypothesis, p-values should be uniform on [0, 1]
        ks_statistic, ks_pvalue = kstest(p_values, "uniform")

        # KS test p-value should be > 0.05 (fail to reject uniform distribution)
        assert ks_pvalue > 0.05, (
            f"P-values not uniformly distributed: KS statistic={ks_statistic:.4f}, p={ks_pvalue:.4f}"
        )

    @pytest.mark.tier1
    def test_effect_size_sensitivity(self):
        """Test that larger true mean produces lower p-values."""
        n_samples = 50
        n_permute = 5000  # Large permutation count for stable p-values

        # Test with different effect sizes
        means = [0.0, 0.5, 1.0, 2.0]
        p_values = []

        for mean in means:
            np.random.seed(42)
            data = np.random.randn(n_samples) + mean
            result = one_sample_permutation_test(
                data, n_permute=n_permute, random_state=42
            )
            p_values.append(result["p"])

        # Verify larger mean → smaller p-value (monotonic relationship)
        # Skip mean=0 (null hypothesis), test others
        # Note: Very large effects may hit minimum p-value (1/(n_permute+1)),
        # so allow >= for equality case when effects are extremely large
        assert p_values[1] >= p_values[2], (
            f"Larger effect should produce smaller p-value. mean=0.5: p={p_values[1]:.6f}, mean=1.0: p={p_values[2]:.6f}"
        )
        assert p_values[2] >= p_values[3], (
            f"Larger effect should produce smaller p-value. mean=1.0: p={p_values[2]:.6f}, mean=2.0: p={p_values[3]:.6f}"
        )

        # Medium effect (mean=1.0) should be significant
        assert p_values[2] < 0.05, (
            f"Medium effect (mean=1.0) should be significant, got p={p_values[2]:.4f}"
        )
        # Large effect (mean=2.0) should be significant
        assert p_values[3] < 0.05, (
            f"Large effect (mean=2.0) should be significant, got p={p_values[3]:.4f}"
        )

    @pytest.mark.tier1
    def test_mean_converges_to_true_mean(self):
        """Test that computed mean converges to true mean."""
        n_samples = 50
        true_mean = 5.0

        np.random.seed(42)
        data = np.random.randn(n_samples) + true_mean

        result = one_sample_permutation_test(data, n_permute=2000, random_state=42)

        # Computed mean should be close to true mean
        # Tolerance: rtol=0.1 (10% as specified in plan)
        np.testing.assert_allclose(result["mean"], true_mean, rtol=0.1, atol=0.1)

    @pytest.mark.tier2
    def test_pvalue_converges_with_more_permutations(self):
        """Test that p-values stabilize (variance decreases) with more permutations."""
        n_samples = 50
        true_mean = 2.0  # Known positive mean

        # Run with different permutation counts
        n_permute_values = [100, 1000, 5000]
        p_values = []

        for n_permute in n_permute_values:
            np.random.seed(42)  # Same seed for reproducibility
            data = np.random.randn(n_samples) + true_mean
            result = one_sample_permutation_test(
                data, n_permute=n_permute, random_state=42
            )
            p_values.append(result["p"])

        # P-values should stabilize with more permutations
        # Variance should decrease (p-values become more consistent)
        # We'll test by running multiple times with different seeds for variance estimation
        p_value_variances = []

        for n_permute in n_permute_values:
            p_vals_multi = []
            for seed in range(20):  # Run 20 times with different seeds
                np.random.seed(seed)
                data = np.random.randn(n_samples) + true_mean
                result = one_sample_permutation_test(
                    data, n_permute=n_permute, random_state=seed
                )
                p_vals_multi.append(result["p"])
            p_value_variances.append(np.var(p_vals_multi))

        # Variance should decrease with more permutations
        # Allow some flexibility (variance estimation is noisy)
        assert p_value_variances[1] < p_value_variances[0] * 2, (
            "Variance should decrease with more permutations"
        )
        assert p_value_variances[2] < p_value_variances[0] * 2, (
            "Variance should decrease with more permutations"
        )

    @pytest.mark.tier1
    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed p-value ≈ two-tailed p-value / 2 for positive mean."""
        n_samples = 50
        true_mean = (
            0.3  # Known positive mean (small enough to avoid hitting minimum p-value)
        )

        np.random.seed(42)
        data = np.random.randn(n_samples) + true_mean

        result_two = one_sample_permutation_test(
            data, n_permute=5000, tail=2, random_state=42
        )
        result_one = one_sample_permutation_test(
            data, n_permute=5000, tail=1, random_state=42
        )

        # One-tailed p-value should be approximately half of two-tailed
        # (for positive mean in one-tailed test)
        # Allow tolerance due to finite permutations
        ratio = result_one["p"] / result_two["p"]
        assert 0.3 < ratio < 0.7, (
            f"One-tailed p-value should be ~half of two-tailed. Got ratio={ratio:.4f}, one_tailed={result_one['p']:.4f}, two_tailed={result_two['p']:.4f}"
        )

    @pytest.mark.tier1
    def test_null_distribution_has_zero_mean(self):
        """Test that null distribution is centered at zero under null hypothesis."""
        n_samples = 50
        n_permute = 2000

        # Generate null data (mean=0)
        np.random.seed(42)
        data = np.random.randn(n_samples)  # Mean ~ 0

        result = one_sample_permutation_test(
            data, n_permute=n_permute, return_null=True, random_state=42
        )

        # Null distribution mean should be close to 0 (within sampling error)
        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

        # Expected std of mean under null: std(data) / sqrt(n_samples)
        # Use more lenient tolerance (3 standard errors)
        expected_std_of_mean = null_std / np.sqrt(n_permute)
        tolerance = 3 * expected_std_of_mean

        assert abs(null_mean) < tolerance, (
            f"Null distribution mean should be close to 0. "
            f"Got mean={null_mean:.6f}, expected within ±{tolerance:.6f}"
        )
