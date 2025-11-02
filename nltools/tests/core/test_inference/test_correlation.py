"""Tests for correlation permutation tests and helper functions."""

import pytest
import numpy as np

from nltools.algorithms.inference.correlation import (
    correlation_permutation_test,
    _pearson_correlation,
)
from nltools.tests.core.test_inference import (
    N_PERMUTE_BACKEND,
    N_PERMUTE_STATS_COMPARISON,
    TOLERANCE_STATS_DETERMINISTIC,
    TOLERANCE_STATS_PVALUE,
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.backends import check_gpu_available
from nltools.stats import correlation_permutation as stats_correlation


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

    def test_basic_functionality_single_feature(self):
        """Test basic correlation test with 1D arrays."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5  # Moderate correlation

        result = correlation_permutation_test(x, y, n_permute=500, random_state=42)

        assert "correlation" in result
        assert "p" in result
        assert "backend" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert isinstance(result["p"], (float, np.floating))
        assert 0 <= result["p"] <= 1
        assert -1 <= result["correlation"] <= 1

    def test_basic_functionality_multi_feature(self):
        """Test correlation test with 2D arrays (feature-wise)."""
        np.random.seed(42)
        data1 = np.random.randn(50, 10)  # 50 samples, 10 features
        data2 = data1 + np.random.randn(50, 10) * 0.3  # Correlated

        result = correlation_permutation_test(
            data1, data2, n_permute=500, random_state=42
        )

        assert result["correlation"].shape == (10,)
        assert result["p"].shape == (10,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert np.all((result["correlation"] >= -1) & (result["correlation"] <= 1))

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5

        result1 = correlation_permutation_test(x, y, n_permute=200, random_state=42)
        result2 = correlation_permutation_test(x, y, n_permute=200, random_state=42)

        np.testing.assert_almost_equal(result1["correlation"], result2["correlation"])
        np.testing.assert_almost_equal(result1["p"], result2["p"])

    def test_return_null_distribution_single(self):
        """Test that null distribution is returned for single feature."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = np.random.randn(50)

        result = correlation_permutation_test(
            x, y, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100,)

    def test_return_null_distribution_multi(self):
        """Test null distribution for multi-feature data."""
        np.random.seed(42)
        data1 = np.random.randn(50, 5)
        data2 = np.random.randn(50, 5)

        result = correlation_permutation_test(
            data1, data2, n_permute=100, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100, 5)

    def test_significant_correlation(self):
        """Test that significant correlation is detected."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Very strong correlation

        result = correlation_permutation_test(x, y, n_permute=1000, random_state=42)

        assert result["p"] < 0.05  # Should be significant
        assert result["correlation"] > 0.9  # Should be strong positive

    def test_non_significant_correlation(self):
        """Test that non-significant correlation has high p-value."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)  # Independent

        result = correlation_permutation_test(x, y, n_permute=1000, random_state=42)

        assert result["p"] > 0.05  # Should not be significant

    def test_spearman_metric(self):
        """Test Spearman correlation metric in permutation test."""
        np.random.seed(42)
        # Create monotonic but non-linear relationship
        x = np.random.randn(100)
        y = x**3 + np.random.randn(100) * 0.1

        result = correlation_permutation_test(
            x, y, n_permute=500, metric="spearman", random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert result["p"] < 0.05  # Should be significant (monotonic)
        assert result["correlation"] > 0.9  # Strong positive Spearman

    def test_kendall_metric(self):
        """Test Kendall correlation metric in permutation test."""
        np.random.seed(42)
        # Create monotonic relationship
        x = np.arange(80)
        y = x + np.random.randn(80) * 5

        result = correlation_permutation_test(
            x, y, n_permute=200, metric="kendall", random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert result["p"] < 0.05  # Should be significant (monotonic)
        assert result["correlation"] > 0.7  # Strong positive Kendall

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

    def test_cpu_parallel_correctness(self):
        """Test CPU parallelization produces correct results."""
        np.random.seed(42)
        data1 = np.random.randn(50, 20)
        data2 = data1 + np.random.randn(50, 20) * 0.5

        result = correlation_permutation_test(
            data1, data2, n_permute=500, backend=None, n_jobs=2, random_state=42
        )

        # Observed correlations should be positive (data2 derived from data1)
        assert np.all(result["correlation"] > 0)

        # P-values should be valid
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert "cpu-parallel" in result["backend"]

    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        x = np.random.randn(50)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            correlation_permutation_test(x, y, tail=3)

    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        x = np.random.randn(5, 5, 5)  # 3D
        y = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="data1 must be 1D or 2D"):
            correlation_permutation_test(x, y)

    def test_mismatched_shapes(self):
        """Test that mismatched shapes raise error."""
        x = np.random.randn(50, 5)  # 5 features
        y = np.random.randn(50, 10)  # 10 features (mismatch!)

        with pytest.raises(ValueError, match="must have same shape"):
            correlation_permutation_test(x, y)

    def test_backend_consistency_single_feature(self, backends):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5

        results = {}
        for backend in backends:
            results[backend] = correlation_permutation_test(
                x, y, n_permute=N_PERMUTE_BACKEND, backend=backend, random_state=42
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

    def test_backend_consistency_multi_feature(self, backends):
        """Test backend consistency for multi-feature data."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data1 = np.random.randn(50, 10)
        data2 = data1 + np.random.randn(50, 10) * 0.3

        results = {}
        for backend in backends:
            results[backend] = correlation_permutation_test(
                data1,
                data2,
                n_permute=N_PERMUTE_BACKEND,
                backend=backend,
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

    def test_matches_stats_py_single_feature(self):
        """Test that results match stats.py for single feature."""
        np.random.seed(42)
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5

        # New implementation
        result_new = correlation_permutation_test(
            x, y, n_permute=N_PERMUTE_STATS_COMPARISON, backend="numpy", random_state=42
        )

        # Old implementation
        result_old = stats_correlation(
            x,
            y,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            method="permute",
            metric="pearson",
            tail=2,
            n_jobs=1,
            random_state=42,
        )

        # Correlation should be identical (deterministic)
        np.testing.assert_allclose(
            result_new["correlation"],
            result_old["correlation"],
            rtol=TOLERANCE_STATS_DETERMINISTIC,
        )
        # P-values will differ slightly due to different random sampling (~15%)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE
        )

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data1 = np.random.randn(50, 1000)
        data2 = data1 + np.random.randn(50, 1000) * 0.3

        # NumPy backend
        result_numpy = correlation_permutation_test(
            data1, data2, n_permute=500, backend="numpy", random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = correlation_permutation_test(
            data1,
            data2,
            n_permute=500,
            backend="torch",
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
            data1, data2, n_permute=200, backend=None, random_state=42
        )

        # GPU vectorized (should process all features simultaneously)
        result_gpu = correlation_permutation_test(
            data1, data2, n_permute=200, backend="torch", random_state=42
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
            backend=None,
            random_state=42,
        )

        # GPU vectorized Spearman
        result_gpu = correlation_permutation_test(
            data1,
            data2,
            n_permute=200,
            metric="spearman",
            backend="torch",
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
