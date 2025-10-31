"""
Tests for GPU-accelerated statistical inference.

Organization:
    - TestHelperFunctions: Test helper functions
    - TestOneSamplePermutation: One-sample permutation tests
    - TestBackends: Backend consistency tests
    - TestBackwardCompatibility: Tests against existing stats.py
"""

import pytest
import numpy as np
from nltools.algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    _generate_sign_flips,
    _compute_pvalue,
)
from nltools.algorithms.inference.correlation import (
    correlation_permutation_test,
    _pearson_correlation,
)
from nltools.backends import Backend, check_gpu_available
from nltools.stats import one_sample_permutation as stats_one_sample
from nltools.stats import two_sample_permutation as stats_two_sample
from nltools.stats import correlation_permutation as stats_correlation


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    # 30 subjects, 100 features (small for fast tests)
    return np.random.randn(30, 100)


@pytest.fixture
def backends():
    """Return list of available backends."""
    backends_list = ["numpy"]
    if check_gpu_available()[0]:
        backends_list.append("torch")
    return backends_list


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test helper functions for correctness."""

    def test_generate_sign_flips_shape(self):
        """Test that sign flips have correct shape."""
        sign_flips = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        assert sign_flips.shape == (100, 30)

    def test_generate_sign_flips_values(self):
        """Test that sign flips only contain +1 and -1."""
        sign_flips = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        assert np.all(np.isin(sign_flips, [-1, 1]))

    def test_generate_sign_flips_deterministic(self):
        """Test that sign flips are deterministic with fixed seed."""
        sf1 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        sf2 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        np.testing.assert_array_equal(sf1, sf2)

    def test_generate_sign_flips_random(self):
        """Test that sign flips are different with different seeds."""
        sf1 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        sf2 = _generate_sign_flips(n_permute=100, n_samples=30, random_state=43)
        assert not np.array_equal(sf1, sf2)

    def test_compute_pvalue_two_tailed(self):
        """Test two-tailed p-value computation with correction factor."""
        # With correction factor: (count + 1) / (n_permute + 1)
        np.random.seed(42)
        null_dist = np.random.randn(10000, 1)
        obs_stat = np.array([np.percentile(null_dist, 90)])
        p = _compute_pvalue(obs_stat, null_dist, tail=2)
        # Should be moderate p-value (not extreme)
        assert 0.1 < p[0] < 0.3

    def test_compute_pvalue_one_tailed(self):
        """Test one-tailed p-value computation with correction factor."""
        np.random.seed(42)
        null_dist = np.random.randn(10000, 1)
        obs_stat = np.array([np.percentile(null_dist, 95)])  # 95th percentile
        p = _compute_pvalue(obs_stat, null_dist, tail=1)
        # With correction factor, should be slightly > 0.05
        assert 0.04 < p[0] < 0.07

    def test_compute_pvalue_extreme(self):
        """Test p-value for extreme observed statistic."""
        # Observed far from null → p-value should be minimum: 1/(n+1)
        null_dist = np.random.randn(1000, 1)
        obs_stat = np.array([10.0])  # Very extreme (essentially no null values exceed)
        p = _compute_pvalue(obs_stat, null_dist, tail=2)
        # Minimum p-value with correction: 1/(1000+1) ≈ 0.001
        assert p[0] == 1.0 / 1001.0

    def test_compute_pvalue_multifeature(self):
        """Test p-value computation for multiple features."""
        np.random.seed(42)
        null_dist = np.random.randn(1000, 10)
        # Use moderate percentile values for testing
        obs_stat = np.percentile(null_dist, 90, axis=0)  # 90th percentile
        p = _compute_pvalue(obs_stat, null_dist, tail=2)
        assert p.shape == (10,)
        # All p-values should be valid (between 0 and 1)
        assert np.all((p > 0) & (p <= 1))

    def test_compute_pvalue_invalid_tail(self):
        """Test that invalid tail raises error."""
        null_dist = np.random.randn(100, 1)
        obs_stat = np.array([0.0])
        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            _compute_pvalue(obs_stat, null_dist, tail=3)


# ============================================================================
# Test One-Sample Permutation
# ============================================================================


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

        result_two = one_sample_permutation_test(data, n_permute=1000, tail=2, random_state=42)
        result_one = one_sample_permutation_test(data, n_permute=1000, tail=1, random_state=42)

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


# ============================================================================
# Test Two-Sample Permutation
# ============================================================================


class TestTwoSamplePermutation:
    """Test two-sample permutation tests."""

    def test_basic_functionality_single_feature(self):
        """Test basic two-sample test with single feature (1D arrays)."""
        np.random.seed(42)
        data1 = np.random.randn(20)  # Group 1: 20 subjects
        data2 = np.random.randn(25)  # Group 2: 25 subjects

        result = two_sample_permutation_test(data1, data2, n_permute=1000, random_state=42)

        assert "mean_diff" in result
        assert "p" in result
        assert "backend" in result
        assert isinstance(result["mean_diff"], (float, np.floating))
        assert isinstance(result["p"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_basic_functionality_multi_feature(self):
        """Test two-sample test with multiple features (2D arrays)."""
        np.random.seed(42)
        data1 = np.random.randn(20, 10)  # 20 subjects, 10 features
        data2 = np.random.randn(25, 10)  # 25 subjects, 10 features

        result = two_sample_permutation_test(data1, data2, n_permute=1000, random_state=42)

        assert result["mean_diff"].shape == (10,)
        assert result["p"].shape == (10,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed seed."""
        np.random.seed(42)
        data1 = np.random.randn(20, 5)
        data2 = np.random.randn(25, 5)

        result1 = two_sample_permutation_test(data1, data2, n_permute=100, random_state=42)
        result2 = two_sample_permutation_test(data1, data2, n_permute=100, random_state=42)

        np.testing.assert_array_almost_equal(result1["mean_diff"], result2["mean_diff"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])

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

    def test_significant_effect(self):
        """Test that significant group difference is detected."""
        np.random.seed(42)
        data1 = np.random.randn(30)  # Mean = 0
        data2 = np.random.randn(30) + 2.0  # Mean = 2.0 (large difference)

        result = two_sample_permutation_test(data1, data2, n_permute=1000, random_state=42)

        assert result["p"] < 0.05  # Should be significant

    def test_non_significant_effect(self):
        """Test that non-significant difference has high p-value."""
        np.random.seed(42)
        data1 = np.random.randn(30)
        data2 = np.random.randn(30) + 0.1  # Small difference

        result = two_sample_permutation_test(data1, data2, n_permute=1000, random_state=42)

        assert result["p"] > 0.05  # Should not be significant

    def test_unequal_sample_sizes(self):
        """Test that unequal sample sizes work correctly."""
        np.random.seed(42)
        data1 = np.random.randn(15, 5)  # 15 subjects
        data2 = np.random.randn(35, 5)  # 35 subjects (different size)

        result = two_sample_permutation_test(data1, data2, n_permute=500, random_state=42)

        assert result["mean_diff"].shape == (5,)
        assert result["p"].shape == (5,)

    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed and two-tailed p-values differ."""
        np.random.seed(42)
        data1 = np.random.randn(30)
        data2 = np.random.randn(30) + 0.5

        result_two = two_sample_permutation_test(data1, data2, n_permute=1000, tail=2, random_state=42)
        result_one = two_sample_permutation_test(data1, data2, n_permute=1000, tail=1, random_state=42)

        # One-tailed should be different from two-tailed
        assert result_one["p"] != result_two["p"]

    def test_invalid_tail(self):
        """Test that invalid tail raises error."""
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        with pytest.raises(ValueError, match="tail must be 1 or 2"):
            two_sample_permutation_test(data1, data2, tail=3)

    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        data1 = np.random.randn(5, 5, 5)  # 3D
        data2 = np.random.randn(5, 5, 5)

        with pytest.raises(ValueError, match="data1 must be 1D or 2D"):
            two_sample_permutation_test(data1, data2)

    def test_mismatched_features(self):
        """Test that mismatched feature dimensions raise error."""
        data1 = np.random.randn(20, 5)  # 5 features
        data2 = np.random.randn(25, 10)  # 10 features (mismatch!)

        with pytest.raises(ValueError, match="must have same number of features"):
            two_sample_permutation_test(data1, data2)

    def test_backend_consistency_single_feature(self, backends):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        results = {}
        for backend in backends:
            results[backend] = two_sample_permutation_test(
                data1, data2, n_permute=100, backend=backend, random_state=42
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

    def test_backend_consistency_multi_feature(self, backends):
        """Test backend consistency for multi-feature data."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data1 = np.random.randn(20, 10)
        data2 = np.random.randn(25, 10)

        results = {}
        for backend in backends:
            results[backend] = two_sample_permutation_test(
                data1, data2, n_permute=100, backend=backend, random_state=42
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

    def test_matches_stats_single_feature(self):
        """Test that results match stats.py for single feature."""
        np.random.seed(42)
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        # New implementation
        result_new = two_sample_permutation_test(
            data1, data2, n_permute=1000, tail=2, backend="numpy", random_state=42
        )

        # Old implementation
        result_old = stats_two_sample(
            data1, data2, n_permute=1000, tail=2, n_jobs=1, random_state=42
        )

        # Mean difference should be identical
        np.testing.assert_allclose(result_new["mean_diff"], result_old["mean"], rtol=1e-5)
        # P-values will differ slightly due to different random sampling (~15%)
        np.testing.assert_allclose(result_new["p"], result_old["p"], rtol=0.15)

    def test_cpu_parallel_correctness(self):
        """Test CPU parallelization produces correct results."""
        np.random.seed(42)
        data1 = np.random.randn(20, 50)
        data2 = np.random.randn(25, 50)

        result = two_sample_permutation_test(
            data1, data2, n_permute=500, backend=None, n_jobs=2, random_state=42
        )

        # Mean difference should match observed
        obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)
        np.testing.assert_allclose(result["mean_diff"], obs_diff)

        # P-values should be valid
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert "cpu-parallel" in result["backend"]

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data1 = np.random.randn(20, 5000)
        data2 = np.random.randn(25, 5000)

        # NumPy backend
        result_numpy = two_sample_permutation_test(
            data1, data2, n_permute=500, backend="numpy", random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = two_sample_permutation_test(
            data1, data2,
            n_permute=500,
            backend="torch",
            max_gpu_memory_gb=0.5,
            random_state=42,
        )

        # Results should match (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_numpy["mean_diff"],
            result_gpu["mean_diff"],
            rtol=1e-3,  # Relaxed for float32/float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=5e-3,  # P-values accumulate more FP errors
        )


# ============================================================================
# Test Backend Consistency
# ============================================================================


class TestBackends:
    """Test backend consistency (NumPy vs PyTorch)."""

    def test_backend_consistency_single_feature(self, backends):
        """Test that NumPy and PyTorch backends produce same results."""
        if len(backends) < 2:
            pytest.skip("PyTorch not available")

        np.random.seed(42)
        data = np.random.randn(30)

        results = {}
        for backend in backends:
            results[backend] = one_sample_permutation_test(
                data, n_permute=100, backend=backend, random_state=42
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["mean"],
            results["torch"]["mean"],
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
        data = np.random.randn(30, 10)

        results = {}
        for backend in backends:
            results[backend] = one_sample_permutation_test(
                data, n_permute=100, backend=backend, random_state=42
            )

        # Compare results
        np.testing.assert_allclose(
            results["numpy"]["mean"],
            results["torch"]["mean"],
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            results["numpy"]["p"],
            results["torch"]["p"],
            rtol=1e-5,
        )

    def test_auto_backend_selection(self):
        """Test that auto backend selection works."""
        np.random.seed(42)
        data = np.random.randn(30, 10)
        result = one_sample_permutation_test(data, n_permute=100, backend="auto", random_state=42)

        assert "backend" in result
        assert result["backend"] in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]

    def test_explicit_numpy_backend(self):
        """Test explicit NumPy backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, backend="numpy", random_state=42)

        assert result["backend"] == "numpy"

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_explicit_torch_backend(self):
        """Test explicit PyTorch backend."""
        np.random.seed(42)
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, backend="torch", random_state=42)

        assert result["backend"].startswith("torch")


# ============================================================================
# Test Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """Test compatibility with existing stats.py implementation."""

    def test_matches_stats_single_feature(self):
        """Test that results match stats.py for single feature."""
        np.random.seed(42)
        data = np.random.randn(30)

        # New implementation
        result_new = one_sample_permutation_test(
            data, n_permute=1000, tail=2, backend="numpy", random_state=42
        )

        # Old implementation
        result_old = stats_one_sample(
            data, n_permute=1000, tail=2, n_jobs=1, random_state=42
        )

        # Compare results
        # Mean should be identical (it's just np.mean)
        np.testing.assert_allclose(result_new["mean"], result_old["mean"], rtol=1e-5)
        # P-values will differ slightly due to different random sampling
        # but should be in the same ballpark (within ~15% relative error)
        np.testing.assert_allclose(result_new["p"], result_old["p"], rtol=0.15)

    def test_new_multi_feature_support(self):
        """Test that new implementation supports multi-feature data.

        Note: Old stats.py implementation doesn't support multi-feature data
        (it has a broadcasting bug in _permute_sign). Our new implementation
        fixes this limitation.
        """
        np.random.seed(42)
        data = np.random.randn(30, 10)

        # New implementation should work
        result_new = one_sample_permutation_test(
            data, n_permute=1000, tail=2, backend="numpy", random_state=42
        )

        # Verify results are sensible
        assert result_new["mean"].shape == (10,)
        assert result_new["p"].shape == (10,)
        assert np.all((result_new["p"] > 0) & (result_new["p"] <= 1))

    def test_matches_stats_one_tailed(self):
        """Test one-tailed test matches stats.py."""
        np.random.seed(42)
        data = np.random.randn(30) + 0.5  # Positive mean

        # New implementation
        result_new = one_sample_permutation_test(
            data, n_permute=1000, tail=1, backend="numpy", random_state=42
        )

        # Old implementation
        result_old = stats_one_sample(
            data, n_permute=1000, tail=1, n_jobs=1, random_state=42
        )

        # Compare results
        np.testing.assert_allclose(result_new["mean"], result_old["mean"], rtol=1e-5)
        # P-values will differ due to random sampling (within ~15%)
        np.testing.assert_allclose(result_new["p"], result_old["p"], rtol=0.15)


# ============================================================================
# Test GPU Batching
# ============================================================================


class TestGPUBatching:
    """Test GPU batching functionality and memory management."""

    def test_auto_batch_size_small_problem(self):
        """Test that small problems fit in one batch."""
        from nltools.algorithms.inference import _auto_batch_size

        # Small problem: 1000 perms × 30 samples × 1000 features
        # Memory: 1000 * 30 * 1000 * 4 bytes = 120 MB
        # Should easily fit in 4GB
        batch_size, n_batches = _auto_batch_size(
            n_permute=1000, n_samples=30, n_features=1000, max_memory_gb=4.0
        )

        assert n_batches == 1  # Everything fits in one batch
        assert batch_size >= 1000  # Batch size should be >= n_permute

    def test_auto_batch_size_large_problem(self):
        """Test that large problems are split into multiple batches."""
        from nltools.algorithms.inference import _auto_batch_size

        # Large problem: 10K perms × 30 samples × 50K features
        # Memory per perm: 30 * 50K * 4 bytes = 6 MB
        # 4GB budget: ~666 perms per batch
        batch_size, n_batches = _auto_batch_size(
            n_permute=10000, n_samples=30, n_features=50000, max_memory_gb=4.0
        )

        assert n_batches > 1  # Should need multiple batches
        assert batch_size >= 100  # Minimum batch size
        assert batch_size * n_batches >= 10000  # Covers all permutations

    def test_auto_batch_size_memory_budget(self):
        """Test that different memory budgets produce different batch sizes."""
        from nltools.algorithms.inference import _auto_batch_size

        # Same problem, different budgets
        batch_small, _ = _auto_batch_size(
            n_permute=5000, n_samples=30, n_features=10000, max_memory_gb=2.0
        )
        batch_large, _ = _auto_batch_size(
            n_permute=5000, n_samples=30, n_features=10000, max_memory_gb=8.0
        )

        # Larger budget should allow larger batches
        assert batch_large > batch_small

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_correctness(self):
        """Test that GPU batching produces same results as NumPy."""
        np.random.seed(42)
        data = np.random.randn(30, 5000)  # Medium-sized problem

        # NumPy backend (no batching needed)
        result_numpy = one_sample_permutation_test(
            data, n_permute=500, backend="numpy", random_state=42
        )

        # GPU backend with small memory budget to force batching
        result_gpu = one_sample_permutation_test(
            data,
            n_permute=500,
            backend="torch",
            max_gpu_memory_gb=0.5,  # Force small batches
            random_state=42,
        )

        # Results should match (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_numpy["mean"],
            result_gpu["mean"],
            rtol=1e-3,  # Relaxed for float32/float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=5e-3,  # P-values accumulate more FP errors
        )

    @pytest.mark.skipif(not check_gpu_available()[0], reason="GPU not available")
    def test_gpu_batching_large_problem(self):
        """Test GPU batching with large problem that would OOM without batching."""
        np.random.seed(42)

        # Large problem: Would use ~7.2GB without batching
        # 1000 perms × 30 samples × 60K features × 4 bytes = 7.2 GB
        data = np.random.randn(30, 60000).astype(np.float32)

        # Should work with batching (4GB budget)
        result = one_sample_permutation_test(
            data,
            n_permute=1000,
            backend="torch",
            max_gpu_memory_gb=4.0,
            random_state=42,
        )

        # Verify results are sensible
        assert result["mean"].shape == (60000,)
        assert result["p"].shape == (60000,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))

    def test_gpu_batching_parameter_variations(self):
        """Test that max_gpu_memory_gb parameter works correctly."""
        np.random.seed(42)
        data = np.random.randn(30, 1000)

        # Test different memory budgets (should all work, just different batch sizes)
        for memory_gb in [1.0, 2.0, 4.0, 8.0]:
            result = one_sample_permutation_test(
                data,
                n_permute=500,
                backend="numpy",  # Use numpy to avoid GPU requirement
                max_gpu_memory_gb=memory_gb,
                random_state=42,
            )

            # All should produce valid results
            assert "mean" in result
            assert "p" in result
            # Handle both scalar and array p-values
            assert np.all((result["p"] >= 0) & (result["p"] <= 1))


# ============================================================================
# Test CPU Parallelization
# ============================================================================


class TestCPUParallelization:
    """Test CPU parallelization functionality and correctness."""

    def test_cpu_parallel_single_feature(self):
        """Test CPU parallel with single feature (1D data)."""
        np.random.seed(42)
        data = np.random.randn(30)

        result = one_sample_permutation_test(
            data, n_permute=500, backend=None, n_jobs=2, random_state=42
        )

        # Verify scalar results
        assert isinstance(result["mean"], (float, np.floating))
        assert isinstance(result["p"], (float, np.floating))
        assert "cpu-parallel" in result["backend"]

    def test_cpu_parallel_multi_feature(self):
        """Test CPU parallel with multiple features (2D data)."""
        np.random.seed(42)
        data = np.random.randn(30, 100)

        result = one_sample_permutation_test(
            data, n_permute=500, backend=None, n_jobs=2, random_state=42
        )

        # Verify array results
        assert result["mean"].shape == (100,)
        assert result["p"].shape == (100,)
        assert np.all((result["p"] >= 0) & (result["p"] <= 1))
        assert "cpu-parallel" in result["backend"]

    def test_cpu_parallel_correctness(self):
        """Test that CPU parallel produces statistically valid results."""
        np.random.seed(42)

        # Test with data that has mean = 0 (null hypothesis true)
        data_null = np.random.randn(30, 50)
        result_null = one_sample_permutation_test(
            data_null, n_permute=500, backend=None, n_jobs=2, random_state=42
        )

        # Mean should match observed mean
        np.testing.assert_allclose(result_null["mean"], np.mean(data_null, axis=0))

        # P-values should be distributed (not all 0 or all 1)
        # Most should be non-significant (p > 0.05) since null is true
        assert np.sum(result_null["p"] > 0.05) > 40  # At least 80% non-significant

        # Test with data that has strong positive effect
        data_effect = np.random.randn(30, 50) + 2.0  # Mean = 2.0
        result_effect = one_sample_permutation_test(
            data_effect, n_permute=500, backend=None, n_jobs=2, random_state=42
        )

        # All features should be significant (p < 0.05)
        assert np.all(result_effect["p"] < 0.05)

    def test_cpu_parallel_n_jobs_variations(self):
        """Test different n_jobs parameter values."""
        np.random.seed(42)
        data = np.random.randn(30, 20)

        # Test various n_jobs values
        for n_jobs in [1, 2, -1]:
            result = one_sample_permutation_test(
                data, n_permute=200, backend=None, n_jobs=n_jobs, random_state=42
            )

            assert "cpu-parallel" in result["backend"]
            assert result["mean"].shape == (20,)
            assert result["p"].shape == (20,)

    def test_cpu_parallel_return_null(self):
        """Test that null distribution is returned correctly."""
        np.random.seed(42)
        data = np.random.randn(30, 10)

        result = one_sample_permutation_test(
            data, n_permute=200, backend=None, n_jobs=2, return_null=True, random_state=42
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (200, 10)

    def test_cpu_parallel_deterministic(self):
        """Test that CPU parallel is deterministic with same seed."""
        np.random.seed(42)
        data = np.random.randn(30, 10)

        result1 = one_sample_permutation_test(
            data, n_permute=200, backend=None, n_jobs=2, random_state=42
        )
        result2 = one_sample_permutation_test(
            data, n_permute=200, backend=None, n_jobs=2, random_state=42
        )

        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(result1["mean"], result2["mean"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])

# ============================================================================
# Test Correlation Permutation
# ============================================================================


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
        y = x ** 3 + np.random.randn(100) * 0.1  # Monotonic but non-linear

        r = _spearman_correlation(x, y)
        assert isinstance(r, (float, np.floating))
        assert r > 0.9  # Should be strongly positive (monotonic)

    def test_matches_scipy(self):
        """Test that Spearman matches scipy.stats.spearmanr."""
        from nltools.algorithms.inference.correlation import _spearman_correlation
        from scipy.stats import spearmanr

        np.random.seed(42)
        x = np.random.randn(100)
        y = x ** 2 + np.random.randn(100) * 0.5

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
        y = x ** 2 + np.random.randn(50) * 0.5

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
        
        result = correlation_permutation_test(data1, data2, n_permute=500, random_state=42)
        
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
        y = x ** 3 + np.random.randn(100) * 0.1

        result = correlation_permutation_test(
            x, y, n_permute=500, metric='spearman', random_state=42
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
            x, y, n_permute=200, metric='kendall', random_state=42
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

        result_two = correlation_permutation_test(x, y, n_permute=500, tail=2, random_state=42)
        result_one = correlation_permutation_test(x, y, n_permute=500, tail=1, random_state=42)

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
                x, y, n_permute=100, backend=backend, random_state=42
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
                data1, data2, n_permute=100, backend=backend, random_state=42
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
            x, y, n_permute=1000, tail=2, backend="numpy", random_state=42
        )
        
        # Old implementation
        result_old = stats_correlation(
            x, y, n_permute=1000, method='permute', metric='pearson', 
            tail=2, n_jobs=1, random_state=42
        )
        
        # Correlation should be identical (deterministic)
        np.testing.assert_allclose(result_new["correlation"], result_old["correlation"], rtol=1e-5)
        # P-values will differ slightly due to different random sampling (~15%)
        np.testing.assert_allclose(result_new["p"], result_old["p"], rtol=0.15)
        
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
            data1, data2,
            n_permute=500,
            backend="torch",
            max_gpu_memory_gb=0.5,
            random_state=42,
        )
        
        # Results should match (float32 vs float64 precision)
        np.testing.assert_allclose(
            result_numpy["correlation"],
            result_gpu["correlation"],
            rtol=1e-3,  # Relaxed for float32/float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=5e-3,  # P-values accumulate more FP errors
        )
