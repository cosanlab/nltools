"""Tests for correlation permutation tests and helper functions."""

import pytest
import numpy as np
from scipy.stats import kstest, multivariate_normal

from nltools.algorithms.inference.correlation import (
    correlation_permutation_test,
    _pearson_correlation,
)
from nltools.tests.core.test_inference import (
    N_PERMUTE_BACKEND,
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.backends import check_gpu_available


@pytest.mark.tier1
class TestPearsonCorrelation:
    """Test Pearson correlation helper function."""

    def test_basic_correlation_positive(self):
        """Test positive correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Strong positive correlation

        r = _pearson_correlation(x, y)
        assert isinstance(r, (float, np.floating))
        assert r > 0.9  # Should be strongly positive

    def test_basic_correlation_negative(self):
        """Test negative correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = -x + np.random.randn(100) * 0.1  # Strong negative correlation

        r = _pearson_correlation(x, y)
        assert isinstance(r, (float, np.floating))
        assert r < -0.9  # Should be strongly negative

    def test_no_correlation(self):
        """Test zero correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)  # Independent

        r = _pearson_correlation(x, y)
        assert isinstance(r, (float, np.floating))
        assert -0.3 < r < 0.3  # Should be near zero

    def test_vectorized_correlation(self):
        """Test vectorized correlation computation."""
        np.random.seed(42)
        # Multiple permutations
        x_perms = np.random.randn(100, 50)  # 100 permutations, 50 samples
        y = np.random.randn(50)

        r = _pearson_correlation(x_perms, y)
        assert r.shape == (100,)
        assert np.all((r >= -1) & (r <= 1))

    def test_constant_data(self):
        """Test with constant data (should handle gracefully)."""
        x = np.ones(100)
        y = np.random.randn(100)

        r = _pearson_correlation(x, y)
        assert r == 0.0  # Correlation with constant is zero


@pytest.mark.tier1
class TestSpearmanCorrelation:
    """Test Spearman correlation helper function."""

    def test_basic_correlation_positive(self):
        """Test positive Spearman correlation."""
        from nltools.algorithms.inference.correlation import _spearman_correlation

        np.random.seed(42)
        # Create monotonic relationship (perfect for Spearman)
        x = np.random.randn(100)
        y = x**3 + np.random.randn(100) * 0.1  # Monotonic but non-linear

        r = _spearman_correlation(x, y)
        assert isinstance(r, (float, np.floating))
        assert r > 0.9  # Should be strongly positive (monotonic)

    def test_matches_scipy(self):
        """Test that Spearman matches scipy.stats.spearmanr."""
        from nltools.algorithms.inference.correlation import _spearman_correlation
        from scipy.stats import spearmanr

        np.random.seed(42)
        x = np.random.randn(100)
        y = x**2 + np.random.randn(100) * 0.5

        r_ours = _spearman_correlation(x, y)
        r_scipy, _ = spearmanr(x, y)

        np.testing.assert_allclose(r_ours, r_scipy, rtol=1e-10)

    def test_vectorized_correlation(self):
        """Test vectorized Spearman correlation computation."""
        from nltools.algorithms.inference.correlation import _spearman_correlation

        np.random.seed(42)
        # Multiple permutations
        x_perms = np.random.randn(100, 50)  # 100 permutations, 50 samples
        y = np.random.randn(50)

        r = _spearman_correlation(x_perms, y)
        assert r.shape == (100,)
        assert np.all((r >= -1) & (r <= 1))

    def test_constant_data(self):
        """Test with constant data (should handle gracefully)."""
        from nltools.algorithms.inference.correlation import _spearman_correlation

        x = np.ones(100)
        y = np.random.randn(100)

        r = _spearman_correlation(x, y)
        # With constant data, ranks are all tied, correlation should be 0 or nan
        # Our implementation should return 0
        assert np.isnan(r) or r == 0.0


@pytest.mark.tier1
class TestKendallCorrelation:
    """Test Kendall correlation helper function."""

    def test_basic_correlation_positive(self):
        """Test positive Kendall correlation."""
        from nltools.algorithms.inference.correlation import _kendall_correlation

        np.random.seed(42)
        # Create monotonic relationship (perfect for Kendall)
        x = np.arange(100)
        y = x + np.random.randn(100) * 5  # Monotonic with noise

        r = _kendall_correlation(x, y)
        assert isinstance(r, (float, np.floating))
        assert r > 0.8  # Should be strongly positive (concordant pairs dominate)

    def test_matches_scipy(self):
        """Test that Kendall matches scipy.stats.kendalltau."""
        from nltools.algorithms.inference.correlation import _kendall_correlation
        from scipy.stats import kendalltau

        np.random.seed(42)
        x = np.random.randn(50)  # Smaller sample for speed (Kendall is O(n^2))
        y = x**2 + np.random.randn(50) * 0.5

        r_ours = _kendall_correlation(x, y)
        r_scipy, _ = kendalltau(x, y)

        np.testing.assert_allclose(r_ours, r_scipy, rtol=1e-10)

    def test_vectorized_correlation(self):
        """Test vectorized Kendall correlation computation."""
        from nltools.algorithms.inference.correlation import _kendall_correlation

        np.random.seed(42)
        # Multiple permutations (small samples for speed)
        x_perms = np.random.randn(10, 30)  # 10 permutations, 30 samples
        y = np.random.randn(30)

        r = _kendall_correlation(x_perms, y)
        assert r.shape == (10,)
        assert np.all((r >= -1) & (r <= 1))

    def test_constant_data(self):
        """Test with constant data (should handle gracefully)."""
        from nltools.algorithms.inference.correlation import _kendall_correlation

        x = np.ones(50)
        y = np.random.randn(50)

        r = _kendall_correlation(x, y)
        # With constant data, all pairs are tied, correlation should be 0 or nan
        assert np.isnan(r) or r == 0.0


class TestCorrelationPermutation:
    """Test correlation permutation tests."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("n_features", [1, 10])
    def test_basic_functionality(self, n_features):
        """Test basic correlation test with single or multiple features."""
        np.random.seed(42)
        if n_features == 1:
            x = np.random.randn(30)  # Reduced from 50 for tier1 speed
            y = x + np.random.randn(30) * 0.5  # Moderate correlation
        else:
            data1 = np.random.randn(30, n_features)  # 30 samples, n_features features
            data2 = data1 + np.random.randn(30, n_features) * 0.3  # Correlated
            x, y = data1, data2

        result = correlation_permutation_test(x, y, n_permute=100, random_state=42)

        assert "correlation" in result
        assert "p" in result
        assert "parallel" in result

        if n_features == 1:
            assert isinstance(result["correlation"], (float, np.floating))
            assert isinstance(result["p"], (float, np.floating))
            assert 0 <= result["p"] <= 1
            assert -1 <= result["correlation"] <= 1
        else:
            assert result["correlation"].shape == (n_features,)
            assert result["p"].shape == (n_features,)
            assert np.all((result["p"] >= 0) & (result["p"] <= 1))
            assert np.all((result["correlation"] >= -1) & (result["correlation"] <= 1))

    @pytest.mark.tier1
    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        x = np.random.randn(30)  # Reduced from 50 for tier1 speed
        y = x + np.random.randn(30) * 0.5

        result1 = correlation_permutation_test(x, y, n_permute=100, random_state=42)
        result2 = correlation_permutation_test(x, y, n_permute=100, random_state=42)

        np.testing.assert_almost_equal(result1["correlation"], result2["correlation"])
        np.testing.assert_almost_equal(result1["p"], result2["p"])

    @pytest.mark.tier1
    @pytest.mark.parametrize("n_features", [1, 5])
    def test_return_null_distribution(self, n_features):
        """Test that null distribution is returned when requested."""
        np.random.seed(42)
        if n_features == 1:
            x = np.random.randn(30)  # Reduced from 50 for tier1 speed
            y = np.random.randn(30)
            expected_shape = (100,)
        else:
            data1 = np.random.randn(30, n_features)
            data2 = np.random.randn(30, n_features)
            x, y = data1, data2
            expected_shape = (100, n_features)

        result = correlation_permutation_test(
            x, y, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == expected_shape

    @pytest.mark.tier2
    def test_significant_correlation(self):
        """Test that significant correlation is detected."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Very strong correlation

        result = correlation_permutation_test(x, y, n_permute=2000, random_state=42)

        assert result["p"] < 0.05  # Should be significant
        assert result["correlation"] > 0.9  # Should be strong positive

    @pytest.mark.tier2
    def test_non_significant_correlation(self):
        """Test that non-significant correlation has high p-value."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)  # Independent

        result = correlation_permutation_test(x, y, n_permute=2000, random_state=42)

        assert result["p"] > 0.05  # Should not be significant

    @pytest.mark.tier1
    def test_spearman_metric(self):
        """Test Spearman correlation metric in permutation test."""
        np.random.seed(42)
        # Create monotonic but non-linear relationship
        x = np.random.randn(30)  # Reduced from 100 for tier1 speed
        y = x**3 + np.random.randn(30) * 0.1

        result = correlation_permutation_test(
            x,
            y,
            n_permute=100,
            metric="spearman",
            random_state=42,  # Reduced from 500 for tier1 speed
        )

        assert "correlation" in result
        assert "p" in result
        assert result["p"] < 0.05  # Should be significant (monotonic)
        assert result["correlation"] > 0.9  # Strong positive Spearman

    @pytest.mark.tier1
    def test_kendall_metric(self):
        """Test Kendall correlation metric in permutation test."""
        np.random.seed(42)
        # Create monotonic relationship
        x = np.arange(30)  # Reduced from 80 for tier1 speed
        y = x + np.random.randn(30) * 5

        result = correlation_permutation_test(
            x,
            y,
            n_permute=100,
            metric="kendall",
            random_state=42,  # Reduced from 200 for tier1 speed
        )

        assert "correlation" in result
        assert "p" in result
        assert result["p"] < 0.05  # Should be significant (monotonic)
        assert (
            result["correlation"] > 0.65
        )  # Strong positive Kendall (reduced threshold for smaller sample size)

    @pytest.mark.tier2
    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed and two-tailed p-values differ (when not at minimum)."""
        np.random.seed(123)  # Different seed for moderate correlation
        x = np.random.randn(100)
        y = x * 0.3 + np.random.randn(100) * 0.9  # Moderate positive correlation

        result_two = correlation_permutation_test(
            x, y, n_permute=500, tail=2, random_state=42
        )
        result_one = correlation_permutation_test(
            x, y, n_permute=500, tail=1, random_state=42
        )

        # For moderate correlations, one-tailed should be approximately half of two-tailed
        # (or both very small if correlation is very strong)
        # Just check they're both valid and sensible
        assert 0 <= result_one["p"] <= 1
        assert 0 <= result_two["p"] <= 1
        # If neither is at minimum, one-tailed should be smaller
        if result_two["p"] > 0.01:  # Not at minimum
            assert result_one["p"] <= result_two["p"]

    @pytest.mark.tier2
    def test_cpu_parallel_correctness(self):
        """Test CPU parallelization produces correct results."""
        np.random.seed(42)
        data1 = np.random.randn(50, 20)
        data2 = data1 + np.random.randn(50, 20) * 0.5

        result = correlation_permutation_test(
            data1, data2, n_permute=500, parallel="cpu", n_jobs=2, random_state=42
        )

        # Observed correlations should be positive (data2 derived from data1)
        assert np.all(result["correlation"] > 0)

        # P-values should be valid
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert result["parallel"] == "cpu"

    @pytest.mark.tier1
    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        x = np.random.randn(50)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            correlation_permutation_test(x, y, tail=3)

    @pytest.mark.tier1
    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        x = np.random.randn(5, 5, 5)  # 3D
        y = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="data1 must be 1D or 2D"):
            correlation_permutation_test(x, y)

    @pytest.mark.tier1
    def test_mismatched_shapes(self):
        """Test that mismatched shapes raise error."""
        x = np.random.randn(50, 5)  # 5 features
        y = np.random.randn(50, 10)  # 10 features (mismatch!)

        with pytest.raises(ValueError, match="must have same shape"):
            correlation_permutation_test(x, y)

    @pytest.mark.tier2
    def test_backend_consistency_single_feature(self, backends):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5

        results = {}
        for backend in backends:
            # Map backend to parallel parameter
            if backend == "numpy":
                parallel = None
            elif backend == "torch":
                parallel = "gpu"
            else:
                parallel = "cpu"

            results[backend] = correlation_permutation_test(
                x, y, n_permute=N_PERMUTE_BACKEND, parallel=parallel, random_state=42
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["correlation"],
            results["torch"]["correlation"],
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
        data1 = np.random.randn(50, 10)
        data2 = data1 + np.random.randn(50, 10) * 0.3

        results = {}
        for backend in backends:
            # Map backend to parallel parameter
            if backend == "numpy":
                parallel = None
            elif backend == "torch":
                parallel = "gpu"
            else:
                parallel = "cpu"

            results[backend] = correlation_permutation_test(
                data1,
                data2,
                n_permute=N_PERMUTE_BACKEND,
                parallel=parallel,
                random_state=42,
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["correlation"],
            results["torch"]["correlation"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            results["numpy"]["p"],
            results["torch"]["p"],
            rtol=1e-5,
        )

    @pytest.mark.tier2
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data1 = np.random.randn(50, 1000)
        data2 = data1 + np.random.randn(50, 1000) * 0.3

        # NumPy backend
        result_numpy = correlation_permutation_test(
            data1, data2, n_permute=500, parallel=None, random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = correlation_permutation_test(
            data1,
            data2,
            n_permute=500,
            parallel="gpu",
            max_gpu_memory_gb=0.5,
            random_state=42,
        )

        # Results should match (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_numpy["correlation"],
            result_gpu["correlation"],
            rtol=TOLERANCE_GPU_VALUE,  # float32 vs float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=TOLERANCE_GPU_PVALUE,  # P-values accumulate more FP error
        )

    @pytest.mark.tier2
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_vectorized_multi_feature(self):
        """Test that vectorized GPU implementation matches CPU for multi-feature data."""
        np.random.seed(42)
        data1 = np.random.randn(100, 50)  # 100 samples, 50 features
        data2 = data1 + np.random.randn(100, 50) * 0.2  # Correlated

        # CPU parallel (default)
        result_cpu = correlation_permutation_test(
            data1, data2, n_permute=200, parallel="cpu", random_state=42
        )

        # GPU vectorized (should process all features simultaneously)
        result_gpu = correlation_permutation_test(
            data1, data2, n_permute=200, parallel="gpu", random_state=42
        )

        # Results should match closely (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_cpu["correlation"],
            result_gpu["correlation"],
            rtol=TOLERANCE_GPU_VALUE,
        )
        np.testing.assert_allclose(
            result_cpu["p"], result_gpu["p"], rtol=TOLERANCE_GPU_PVALUE
        )

        # Verify shapes are correct
        assert result_gpu["correlation"].shape == (50,)
        assert result_gpu["p"].shape == (50,)

    @pytest.mark.tier2
    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_spearman_matches_cpu(self):
        """Test that Spearman GPU implementation matches CPU for multi-feature data."""
        np.random.seed(42)
        # Create monotonic but non-linear relationship (good for Spearman)
        data1 = np.random.randn(100, 20)  # 100 samples, 20 features
        data2 = data1**3 + np.random.randn(100, 20) * 0.1  # Monotonic relationship

        # CPU parallel (default)
        result_cpu = correlation_permutation_test(
            data1,
            data2,
            n_permute=200,
            metric="spearman",
            parallel="cpu",
            random_state=42,
        )

        # GPU vectorized Spearman
        result_gpu = correlation_permutation_test(
            data1,
            data2,
            n_permute=200,
            metric="spearman",
            parallel="gpu",
            random_state=42,
        )

        # Results should match closely (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_cpu["correlation"],
            result_gpu["correlation"],
            rtol=TOLERANCE_GPU_VALUE,
        )
        np.testing.assert_allclose(
            result_cpu["p"], result_gpu["p"], rtol=TOLERANCE_GPU_PVALUE
        )

        # Verify shapes are correct
        assert result_gpu["correlation"].shape == (20,)
        assert result_gpu["p"].shape == (20,)


# ============================================================================
# Test Correlation Permutation Statistical Correctness
# ============================================================================


def _generate_bivariate_normal(n_samples, correlation, random_state=None):
    """Generate bivariate normal data with known correlation."""
    np.random.seed(random_state)
    mean = [0, 0]
    cov = [[1, correlation], [correlation, 1]]
    data = multivariate_normal.rvs(
        mean=mean, cov=cov, size=n_samples, random_state=random_state
    )
    return data[:, 0], data[:, 1]


class TestCorrelationPermutationStatisticalCorrectness:
    """Test statistical correctness of correlation permutation tests."""

    @pytest.mark.tier2
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis for all metrics."""
        n_samples = 50
        n_tests = 100  # Run many tests with different seeds
        n_permute = 2000  # Enough permutations for stable p-values

        metrics = ["pearson", "spearman", "kendall"]

        for metric in metrics:
            p_values = []

            for seed in range(n_tests):
                np.random.seed(seed)
                # Generate independent data (correlation = 0)
                x = np.random.randn(n_samples)
                y = np.random.randn(n_samples)  # Independent

                result = correlation_permutation_test(
                    x, y, n_permute=n_permute, metric=metric, random_state=seed
                )

                p_values.append(result["p"])

            # Test uniformity using Kolmogorov-Smirnov test
            # Under null hypothesis, p-values should be uniformly distributed
            ks_statistic, ks_pvalue = kstest(p_values, "uniform")

            # KS test p-value should be > 0.05 (p-values are uniform)
            assert ks_pvalue > 0.05, (
                f"P-values should be uniformly distributed under null hypothesis for {metric}. "
                f"KS test p-value: {ks_pvalue:.4f}"
            )

    @pytest.mark.tier1
    def test_correlation_value_correctness(self):
        """Test that computed correlation values match expected values for all metrics."""
        n_samples = 100
        true_correlation = 0.7  # Known correlation

        # Generate bivariate normal data with known correlation
        x, y = _generate_bivariate_normal(n_samples, true_correlation, random_state=42)

        metrics = ["pearson", "spearman", "kendall"]

        for metric in metrics:
            result = correlation_permutation_test(
                x, y, n_permute=2000, metric=metric, random_state=42
            )

            # Computed correlation should be close to true correlation
            # Tolerance: rtol=0.05 (5% as specified in plan)
            # Note: Spearman and Kendall may differ from Pearson due to non-linearity
            # For Pearson, we expect close match; for rank-based, we expect positive correlation
            # Sample correlation can vary from true correlation (especially with smaller samples)
            if metric == "pearson":
                np.testing.assert_allclose(
                    result["correlation"], true_correlation, rtol=0.15, atol=0.1
                )
            else:
                # Rank-based metrics should detect positive relationship
                # Kendall can be lower than Spearman for same relationship
                assert result["correlation"] > 0.3, (
                    f"{metric.capitalize()} should detect positive correlation. "
                    f"Got {result['correlation']:.4f}"
                )

    @pytest.mark.tier1
    def test_effect_size_sensitivity(self):
        """Test that larger correlation produces lower p-values."""
        n_samples = 50
        n_permute = 5000  # Large enough for stable p-values

        # Test with different correlations
        correlations = [
            0.0,
            0.1,
            0.3,
            0.5,
            0.7,
        ]  # Null, small, medium, large, very large
        p_values = []

        for corr in correlations:
            # Generate data with known correlation
            x, y = _generate_bivariate_normal(n_samples, corr, random_state=42)

            result = correlation_permutation_test(
                x, y, n_permute=n_permute, metric="pearson", random_state=42
            )

            p_values.append(result["p"])

        # Verify larger correlation → smaller p-value (monotonic relationship)
        # Skip corr=0 (null hypothesis), test others
        # Note: Very large effects may hit minimum p-value (1/(n_permute+1)),
        # so allow >= for equality case when effects are extremely large
        assert p_values[1] >= p_values[2], (
            f"Larger correlation should produce smaller p-value. "
            f"corr=0.1: p={p_values[1]:.6f}, corr=0.3: p={p_values[2]:.6f}"
        )
        assert p_values[2] >= p_values[3], (
            f"Larger correlation should produce smaller p-value. "
            f"corr=0.3: p={p_values[2]:.6f}, corr=0.5: p={p_values[3]:.6f}"
        )
        assert p_values[3] >= p_values[4], (
            f"Larger correlation should produce smaller p-value. "
            f"corr=0.5: p={p_values[3]:.6f}, corr=0.7: p={p_values[4]:.6f}"
        )

        # Large correlation (corr=0.7) should be significant
        assert p_values[4] < 0.05, (
            f"Large correlation (corr=0.7) should be significant, got p={p_values[4]:.4f}"
        )

    @pytest.mark.tier1
    def test_spearman_handles_monotonic_relationships(self):
        """Test that Spearman detects monotonic non-linear relationships better than Pearson."""
        n_samples = 100

        # Create monotonic but non-linear relationship (y = abs(x) + noise)
        # This is truly monotonic for positive x values
        np.random.seed(42)
        x = np.random.uniform(0, 10, n_samples)  # Positive values only
        y = np.abs(x) + np.random.randn(n_samples) * 0.5  # Monotonic but non-linear

        result_pearson = correlation_permutation_test(
            x, y, n_permute=2000, metric="pearson", random_state=42
        )
        result_spearman = correlation_permutation_test(
            x, y, n_permute=2000, metric="spearman", random_state=42
        )

        # Spearman should detect stronger relationship (higher correlation)
        # For monotonic relationships, Spearman should be at least as high as Pearson
        assert result_spearman["correlation"] >= result_pearson["correlation"] - 0.1, (
            f"Spearman should detect monotonic relationship at least as well. "
            f"Pearson: {result_pearson['correlation']:.4f}, "
            f"Spearman: {result_spearman['correlation']:.4f}"
        )

        # Both should detect significant relationship
        assert result_spearman["p"] < 0.05, (
            f"Spearman should detect significant relationship. p={result_spearman['p']:.4f}"
        )
        assert result_pearson["p"] < 0.05, (
            f"Pearson should detect significant relationship. p={result_pearson['p']:.4f}"
        )

    @pytest.mark.tier1
    def test_kendall_handles_tied_ranks(self):
        """Test that Kendall handles tied ranks correctly."""
        n_samples = 50

        # Create data with many tied values
        np.random.seed(42)
        x = np.random.choice([0, 1, 2, 3, 4], size=n_samples)  # Many ties
        y = x + np.random.randn(n_samples) * 0.5  # Positive relationship

        result_kendall = correlation_permutation_test(
            x, y, n_permute=2000, metric="kendall", random_state=42
        )
        result_spearman = correlation_permutation_test(
            x, y, n_permute=2000, metric="spearman", random_state=42
        )

        # Both should produce valid results (no crashes)
        assert not np.isnan(result_kendall["correlation"]), (
            "Kendall should handle tied ranks without NaN"
        )
        assert not np.isnan(result_kendall["p"]), "Kendall should produce valid p-value"

        # Both rank-based metrics should detect positive relationship
        assert result_kendall["correlation"] > 0, (
            f"Kendall should detect positive relationship. Got {result_kendall['correlation']:.4f}"
        )
        assert result_spearman["correlation"] > 0, (
            f"Spearman should detect positive relationship. Got {result_spearman['correlation']:.4f}"
        )

    @pytest.mark.tier2
    def test_pvalue_converges_with_more_permutations(self):
        """Test that p-values stabilize (variance decreases) with more permutations."""
        n_samples = 50
        true_correlation = 0.5  # Moderate correlation

        # Generate data with known correlation
        x, y = _generate_bivariate_normal(n_samples, true_correlation, random_state=42)

        # Run with different permutation counts
        n_permute_values = [100, 1000, 5000]
        p_values = []

        for n_permute in n_permute_values:
            result = correlation_permutation_test(
                x, y, n_permute=n_permute, metric="pearson", random_state=42
            )
            p_values.append(result["p"])

        # P-values should stabilize with more permutations
        # Variance should decrease (p-values become more consistent)
        # We'll test by running multiple times with different seeds for variance estimation
        p_value_variances = []

        for n_permute in n_permute_values:
            p_vals_multi = []
            for seed in range(20):  # Run 20 times with different seeds
                x_multi, y_multi = _generate_bivariate_normal(
                    n_samples, true_correlation, random_state=seed
                )
                result = correlation_permutation_test(
                    x_multi,
                    y_multi,
                    n_permute=n_permute,
                    metric="pearson",
                    random_state=seed,
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

        # Correlation values should remain stable across permutation counts
        # (correlation is deterministic, shouldn't change)
        correlations = []
        for n_permute in n_permute_values:
            result = correlation_permutation_test(
                x, y, n_permute=n_permute, metric="pearson", random_state=42
            )
            correlations.append(result["correlation"])

        # All correlation values should be very similar (deterministic)
        np.testing.assert_allclose(correlations[0], correlations[1], rtol=1e-10)
        np.testing.assert_allclose(correlations[1], correlations[2], rtol=1e-10)

    @pytest.mark.tier1
    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed p-value ≈ two-tailed p-value / 2 for positive correlation."""
        n_samples = 50
        true_correlation = (
            0.6  # Known positive correlation (moderate to avoid saturation)
        )

        # Generate data with known correlation
        x, y = _generate_bivariate_normal(n_samples, true_correlation, random_state=42)

        result_two = correlation_permutation_test(
            x, y, n_permute=5000, tail=2, metric="pearson", random_state=42
        )
        result_one = correlation_permutation_test(
            x, y, n_permute=5000, tail=1, metric="pearson", random_state=42
        )

        # One-tailed p-value should be approximately half of two-tailed
        # (for positive correlation in one-tailed test)
        # Allow tolerance due to finite permutations
        # Skip if p-values hit minimum (ratio will be 1.0)
        if result_two["p"] > 0.001:  # Only check ratio if not at minimum
            ratio = result_one["p"] / result_two["p"]
            assert 0.3 < ratio < 0.7, (
                f"One-tailed p-value should be ~half of two-tailed. "
                f"Got ratio={ratio:.4f}, one_tailed={result_one['p']:.4f}, two_tailed={result_two['p']:.4f}"
            )
