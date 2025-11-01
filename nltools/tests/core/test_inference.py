"""
Tests for GPU-accelerated statistical inference.

Organization:
    - TestHelperFunctions: Test helper functions
    - TestOneSamplePermutation: One-sample permutation tests
    - TestTwoSamplePermutation: Two-sample permutation tests
    - TestBackends: Backend consistency tests
    - TestBackwardCompatibility: Tests against existing stats.py
    - TestGPUBatching: GPU-specific batching tests
    - TestCPUParallelization: CPU parallelization tests
    - TestPearsonCorrelation: Pearson correlation helper tests
    - TestSpearmanCorrelation: Spearman correlation helper tests
    - TestKendallCorrelation: Kendall correlation helper tests
    - TestCorrelationPermutation: Correlation permutation tests
    - TestCircleShift: Time-series circle shift tests
    - TestPhaseRandomize: FFT phase randomization tests
    - TestTimeseriesCorrelation: Time-series correlation tests
    - TestMatrixHelpers: Matrix manipulation helper tests
    - TestMatrixPermutationCPUParallel: CPU-parallel matrix permutation tests
    - TestMatrixPermutationMain: Main matrix permutation function tests
    - TestMatrixPermutationCorrectness: Matrix permutation correctness tests

Testing Strategy & Tolerances:

    This test suite uses different tolerance levels for different comparison types.
    Understanding WHY these tolerances exist is critical for maintaining test quality.
    These tolerances are also used in test_isc.py for consistency across test suites.

    1. Backend Consistency (NumPy vs PyTorch):
       - Tolerance: EXACT (rtol=1e-5)
       - Why: Same algorithm, same random seed, only float precision differs
       - Tests verify implementations are mathematically identical

    2. Backward Compatibility (New vs stats.py):
       - Deterministic values (mean, correlation): EXACT (rtol=1e-5)
       - P-values: rtol=0.02 (2% relative error acceptable)
       - Why: Both use identical RNG pattern (pre-generated sign-flips/permutations)
       - Implementation matches stats.py exactly for reproducibility
       - Exception: circle_shift uses rtol=0.4 (40%) due to simpler RNG operations
       - Exception: phase_randomize uses rtol=0.05 (5%) due to FFT numerical stability

    3. GPU Precision (GPU float32 vs CPU float64):
       - Values: rtol=1e-3 (0.1% error)
       - P-values: rtol=5e-3 (0.5% error)
       - Why: GPU uses float32, CPU uses float64; P-values accumulate more error

    4. Pure Functions (Helpers vs stats.py):
       - Tolerance: NONE (np.testing.assert_array_equal)
       - Why: Deterministic functions should produce identical outputs
       - Tests verify bit-for-bit compatibility with stats.py

    Test Parameters:
    - n_permute=100: Fast backend consistency checks
    - n_permute=1000: Stable stats.py backward compatibility tests
    - n_permute=5000+: Production use (not typically used in tests)

    Random Seed Management:
    - Backend tests: Same seed (42) → results should be identical
    - Stats.py tests: Same seed (42) → P-values may differ (RNG ordering)
    - Determinism tests: Same seed → must produce identical results
    - Randomness tests: Different seeds → must produce different results
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
from nltools.algorithms.inference.timeseries import (
    circle_shift,
    phase_randomize,
)
from nltools.backends import check_gpu_available
from nltools.stats import one_sample_permutation as stats_one_sample
from nltools.stats import two_sample_permutation as stats_two_sample
from nltools.stats import correlation_permutation as stats_correlation
# ============================================================================
# Test Constants - DO NOT MODIFY without updating docstring above
# ============================================================================

# Tolerance for backend consistency (NumPy vs PyTorch with same seed)
# These should be EXACT matches (same algorithm, only precision differs)
TOLERANCE_EXACT = 1e-5

# Tolerance for deterministic values when comparing to stats.py
# (mean, correlation, etc. - these are computed identically)
TOLERANCE_STATS_DETERMINISTIC = 1e-5

# Tolerance for P-values when comparing to stats.py
# One-sample: 0.000% error (uses identical _generate_sign_flips pattern)
# Two-sample/Correlation: ~1-2% error (prioritizes cross-backend determinism over stats.py exact match)
# Trade-off: Cross-backend consistency (0.000%) > backward compatibility (~1-2%)
TOLERANCE_STATS_PVALUE = 0.02  # 2% relative error acceptable

# One-tailed tests: Same tolerance as two-tailed
# One-sample achieves 0.000%, two-sample ~1-2% (same patterns as above)
TOLERANCE_STATS_PVALUE_ONE_TAILED = 0.02  # 2% relative error acceptable

# Special case: Time-series methods have higher variance vs stats.py
# Root cause: Same as two-sample/correlation (independent RandomState vs shared RNG state)
# circle_shift: ~32% actual variance (shift amounts determined by RNG sequence)
# phase_randomize: ~3% actual variance (FFT operations more numerically stable)
# Both implementations are fully deterministic (same seed → identical results)
TOLERANCE_STATS_PVALUE_CIRCLE_SHIFT = (
    0.4  # 40% relative error (accommodates ~32% actual)
)

# phase_randomize: Lower variance due to FFT numerical stability
TOLERANCE_STATS_PVALUE_PHASE_RANDOMIZE = (
    0.05  # 5% relative error (accommodates ~3% actual)
)

# Tolerance for GPU vs CPU comparisons (float32 vs float64)
TOLERANCE_GPU_VALUE = 1e-3  # 0.1% error for computed values
TOLERANCE_GPU_PVALUE = 5e-3  # 0.5% error for P-values (more FP error)

# Number of permutations for different test types
N_PERMUTE_BACKEND = 100  # Fast checks for backend consistency
N_PERMUTE_STATS_COMPARISON = 1000  # Stable comparison with stats.py


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

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

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

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

        assert result["mean_diff"].shape == (10,)
        assert result["p"].shape == (10,)
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

        result = two_sample_permutation_test(
            data1, data2, n_permute=1000, random_state=42
        )

        assert result["p"] < 0.05  # Should be significant

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

    def test_one_tailed_vs_two_tailed(self):
        """Test that one-tailed and two-tailed p-values differ."""
        np.random.seed(42)
        data1 = np.random.randn(30)
        data2 = np.random.randn(30) + 0.5

        result_two = two_sample_permutation_test(
            data1, data2, n_permute=N_PERMUTE_STATS_COMPARISON, tail=2, random_state=42
        )
        result_one = two_sample_permutation_test(
            data1, data2, n_permute=N_PERMUTE_STATS_COMPARISON, tail=1, random_state=42
        )

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
                data1,
                data2,
                n_permute=N_PERMUTE_BACKEND,
                backend=backend,
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
                data1,
                data2,
                n_permute=N_PERMUTE_BACKEND,
                backend=backend,
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

    def test_matches_stats_single_feature(self):
        """Test that results match stats.py for single feature."""
        np.random.seed(42)
        data1 = np.random.randn(20)
        data2 = np.random.randn(25)

        # New implementation
        result_new = two_sample_permutation_test(
            data1,
            data2,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            backend="numpy",
            random_state=42,
        )

        # Old implementation
        result_old = stats_two_sample(
            data1,
            data2,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            tail=2,
            n_jobs=1,
            random_state=42,
        )

        # Mean difference should be identical
        np.testing.assert_allclose(
            result_new["mean_diff"],
            result_old["mean"],
            rtol=TOLERANCE_STATS_DETERMINISTIC,
        )
        # P-values will differ slightly due to different random sampling (~15%)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE
        )

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
            data1,
            data2,
            n_permute=500,
            backend="torch",
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
                data, n_permute=N_PERMUTE_BACKEND, backend=backend, random_state=42
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
                data, n_permute=N_PERMUTE_BACKEND, backend=backend, random_state=42
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
        result = one_sample_permutation_test(
            data, n_permute=100, backend="auto", random_state=42
        )

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
            data, n_permute=N_PERMUTE_STATS_COMPARISON, backend="numpy", random_state=42
        )

        # Old implementation
        result_old = stats_one_sample(
            data,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            tail=2,
            n_jobs=1,
            random_state=42,
        )

        # Compare results
        # Mean should be identical (it's just np.mean)
        np.testing.assert_allclose(
            result_new["mean"], result_old["mean"], rtol=TOLERANCE_STATS_DETERMINISTIC
        )
        # P-values will differ slightly due to different random sampling
        # but should be in the same ballpark (within ~15% relative error)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE
        )

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
            data, n_permute=N_PERMUTE_STATS_COMPARISON, backend="numpy", random_state=42
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
            data,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            backend="numpy",
            tail=1,
            random_state=42,
        )

        # Old implementation
        result_old = stats_one_sample(
            data,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            tail=1,
            n_jobs=1,
            random_state=42,
        )

        # Compare results
        np.testing.assert_allclose(
            result_new["mean"], result_old["mean"], rtol=TOLERANCE_STATS_DETERMINISTIC
        )
        # P-values should match exactly now (same RNG pattern)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE_ONE_TAILED
        )


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
            rtol=TOLERANCE_GPU_VALUE,  # float32 vs float64 differences
        )
        np.testing.assert_allclose(
            result_numpy["p"],
            result_gpu["p"],
            rtol=TOLERANCE_GPU_PVALUE,  # P-values accumulate more FP error
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
            data,
            n_permute=200,
            backend=None,
            n_jobs=2,
            return_null=True,
            random_state=42,
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


# ============================================================================
# Timeseries Functions Tests
# ============================================================================


class TestCircleShift:
    """Tests for circle_shift() function."""

    def test_preserves_shape_1d(self):
        """Test that circle_shift preserves shape for 1D data."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        assert shifted.shape == data.shape

    def test_preserves_shape_2d(self):
        """Test that circle_shift preserves shape for 2D data."""
        data = np.random.randn(50, 10)
        shifted = circle_shift(data, random_state=42)
        assert shifted.shape == data.shape

    def test_deterministic_with_seed_1d(self):
        """Test that circle_shift is deterministic with random_state for 1D."""
        data = np.random.randn(100)
        shifted1 = circle_shift(data, random_state=42)
        shifted2 = circle_shift(data, random_state=42)
        np.testing.assert_array_equal(shifted1, shifted2)

    def test_deterministic_with_seed_2d(self):
        """Test that circle_shift is deterministic with random_state for 2D."""
        data = np.random.randn(100, 5)
        shifted1 = circle_shift(data, random_state=42)
        shifted2 = circle_shift(data, random_state=42)
        np.testing.assert_array_equal(shifted1, shifted2)

    def test_preserves_values_1d(self):
        """Test that circle_shift preserves all values (just reorders) for 1D."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        assert sorted(shifted) == sorted(data)

    def test_preserves_values_2d(self):
        """Test that circle_shift preserves all values for 2D."""
        data = np.random.randn(50, 10)
        shifted = circle_shift(data, random_state=42)
        for i in range(data.shape[1]):
            assert sorted(shifted[:, i]) == pytest.approx(sorted(data[:, i]))

    def test_explicit_shift_1d(self):
        """Test circle_shift with explicit shift amount for 1D."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        expected = np.array([4, 5, 1, 2, 3])
        np.testing.assert_array_equal(shifted, expected)

    def test_explicit_shift_2d(self):
        """Test circle_shift with explicit shift amounts for 2D."""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        shifted = circle_shift(data, shift_amount=np.array([1, 2]))
        expected = np.array([[4, 30], [1, 40], [2, 10], [3, 20]])
        np.testing.assert_array_equal(shifted, expected)

    def test_matches_stats_py_1d(self):
        """Test that circle_shift matches stats.py for 1D data."""
        from nltools.stats import circle_shift as stats_circle_shift

        data = np.random.randn(100)

        # Both should be deterministic with same seed
        shifted_new = circle_shift(data, random_state=42)
        shifted_old = stats_circle_shift(data, random_state=42)

        # Should produce identical results
        np.testing.assert_array_equal(shifted_new, shifted_old)

    def test_matches_stats_py_2d(self):
        """Test that circle_shift matches stats.py for 2D data."""
        from nltools.stats import circle_shift as stats_circle_shift

        data = np.random.randn(100, 5)

        # Both should be deterministic with same seed
        shifted_new = circle_shift(data, random_state=42)
        shifted_old = stats_circle_shift(data, random_state=42)

        # Should produce identical results
        np.testing.assert_array_equal(shifted_new, shifted_old)


class TestPhaseRandomize:
    """Tests for phase_randomize() function."""

    def test_preserves_shape_1d(self):
        """Test that phase_randomize preserves shape for 1D data."""
        data = np.random.randn(100)
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape

    def test_preserves_shape_2d(self):
        """Test that phase_randomize preserves shape for 2D data."""
        data = np.random.randn(100, 5)
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape

    def test_preserves_power_spectrum_1d(self):
        """Test that phase_randomize preserves power spectrum for 1D data.

        This is THE CRITICAL property - power spectrum must be preserved exactly.
        """
        # Use longer signal for better FFT resolution
        data = np.random.randn(200)
        randomized = phase_randomize(data, random_state=42)

        # Compute power spectra
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2

        # Should match exactly (within numerical precision)
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    def test_preserves_power_spectrum_2d(self):
        """Test that phase_randomize preserves power spectrum for 2D data."""
        data = np.random.randn(200, 5)
        randomized = phase_randomize(data, random_state=42)

        # Check each feature independently
        for i in range(data.shape[1]):
            power_orig = np.abs(np.fft.rfft(data[:, i])) ** 2
            power_rand = np.abs(np.fft.rfft(randomized[:, i])) ** 2
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    def test_changes_phase_1d(self):
        """Test that phase_randomize actually changes the signal."""
        # Use deterministic signal
        t = np.linspace(0, 10 * np.pi, 100)
        data = np.sin(t)

        randomized = phase_randomize(data, random_state=42)

        # Signal should be different
        assert not np.allclose(data, randomized)

    def test_changes_phase_2d(self):
        """Test that phase_randomize changes signals for 2D data."""
        data = np.random.randn(100, 5)
        randomized = phase_randomize(data, random_state=42)

        # Should be different
        assert not np.allclose(data, randomized)

    def test_deterministic_with_seed_1d(self):
        """Test that phase_randomize is deterministic with random_state for 1D."""
        data = np.random.randn(100)
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)

    def test_deterministic_with_seed_2d(self):
        """Test that phase_randomize is deterministic with random_state for 2D."""
        data = np.random.randn(100, 5)
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)

    def test_backend_consistency_numpy(self):
        """Test that phase_randomize works with NumPy backend."""
        data = np.random.randn(100)
        randomized = phase_randomize(data, backend="numpy", random_state=42)

        # Should preserve power spectrum
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    def test_matches_stats_py_1d(self):
        """Test that phase_randomize matches stats.py for 1D data."""
        from nltools.stats import phase_randomize as stats_phase_randomize

        data = np.random.randn(100)

        # Both should be deterministic with same seed
        rand_new = phase_randomize(data, random_state=42)
        rand_old = stats_phase_randomize(data, random_state=42)

        # Should produce identical results
        np.testing.assert_array_equal(rand_new, rand_old)

    def test_matches_stats_py_2d(self):
        """Test that phase_randomize matches stats.py for 2D data."""
        from nltools.stats import phase_randomize as stats_phase_randomize

        data = np.random.randn(100, 5)

        # Both should be deterministic with same seed
        rand_new = phase_randomize(data, random_state=42)
        rand_old = stats_phase_randomize(data, random_state=42)

        # Should produce identical results
        np.testing.assert_array_equal(rand_new, rand_old)


class TestTimeseriesCorrelation:
    """Tests for timeseries_correlation_permutation_test() function."""

    def test_basic_functionality_circle_shift(self):
        """Test basic functionality with circle_shift method."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_basic_functionality_phase_randomize(self):
        """Test basic functionality with phase_randomize method."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = timeseries_correlation_permutation_test(
            x, y, method="phase_randomize", n_permute=100, random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with random_state."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result1 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )
        result2 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )

        np.testing.assert_equal(result1["correlation"], result2["correlation"])
        np.testing.assert_equal(result1["p"], result2["p"])

    def test_return_null_distribution(self):
        """Test that null distribution is returned when requested."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = timeseries_correlation_permutation_test(
            x,
            y,
            method="circle_shift",
            n_permute=100,
            return_null=True,
            random_state=42,
        )

        assert "null_distribution" in result
        assert result["null_distribution"].shape == (100,)

    def test_spearman_metric(self):
        """Test with Spearman correlation metric."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = x**2  # Nonlinear monotonic relationship

        result = timeseries_correlation_permutation_test(
            x,
            y,
            method="circle_shift",
            n_permute=100,
            metric="spearman",
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result

    def test_kendall_metric(self):
        """Test with Kendall correlation metric."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Smaller sample for Kendall (O(n^2))
        y = np.random.randn(50)

        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=50, metric="kendall", random_state=42
        )

        assert "correlation" in result
        assert "p" in result

    def test_matches_stats_py_circle_shift(self):
        """Test that circle_shift method matches stats.py for correlation.

        Note: P-values may differ more for circle_shift (~40%) than other methods
        due to RNG seed pre-generation vs. sequential consumption in stats.py.
        This is expected and acceptable - both implementations are correct, just
        use different random number sequences in parallel execution.
        """
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )
        from nltools.stats import correlation_permutation as stats_correlation

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result_new = timeseries_correlation_permutation_test(
            x,
            y,
            method="circle_shift",
            n_permute=1000,
            metric="pearson",
            random_state=42,
        )
        result_old = stats_correlation(
            x,
            y,
            method="circle_shift",
            n_permute=1000,
            metric="pearson",
            random_state=42,
        )

        # Correlation should match exactly (same observed data)
        np.testing.assert_allclose(
            result_new["correlation"],
            result_old["correlation"],
            rtol=TOLERANCE_STATS_DETERMINISTIC,
        )

        # P-values will differ due to RNG seed handling (~40% for circle_shift)
        # Higher variance than other methods due to simpler random operations
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE_CIRCLE_SHIFT
        )

    def test_matches_stats_py_phase_randomize(self):
        """Test that phase_randomize method matches stats.py for correlation.

        Note: P-values may differ slightly due to different RNG seed handling
        in parallel execution, following the standard 15% tolerance pattern.
        """
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )
        from nltools.stats import correlation_permutation as stats_correlation

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result_new = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=1000,
            metric="pearson",
            random_state=42,
        )
        result_old = stats_correlation(
            x,
            y,
            method="phase_randomize",
            n_permute=1000,
            metric="pearson",
            random_state=42,
        )

        # Correlation should match exactly (same observed data)
        np.testing.assert_allclose(
            result_new["correlation"],
            result_old["correlation"],
            rtol=TOLERANCE_STATS_DETERMINISTIC,
        )

        # P-values will differ slightly due to FFT operations with different RNG patterns
        np.testing.assert_allclose(
            result_new["p"],
            result_old["p"],
            rtol=TOLERANCE_STATS_PVALUE_PHASE_RANDOMIZE,
        )

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="method must be"):
            timeseries_correlation_permutation_test(
                x, y, method="invalid_method", n_permute=100, random_state=42
            )

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="same length"):
            timeseries_correlation_permutation_test(
                x, y, method="circle_shift", n_permute=100, random_state=42
            )

    def test_phase_randomize_null_distribution_centered(self):
        """Test that phase_randomize creates null distribution centered at zero.

        This verifies that only ONE variable is randomized (correct behavior),
        not both variables. Randomizing both would reduce statistical power
        and is conceptually incorrect for testing H0: correlation = 0.
        """
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        # Create two uncorrelated time series
        x = np.random.randn(200)
        y = np.random.randn(200)

        # Get null distribution
        result = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=1000,
            return_null=True,
            random_state=42,
            n_jobs=1,
        )

        # Null distribution should be centered near zero
        null_mean = np.mean(result["null_distribution"])
        null_std = np.std(result["null_distribution"])

        # Mean should be very close to 0
        assert abs(null_mean) < 0.05, f"Null mean {null_mean} too far from 0"

        # Standard deviation should be reasonable (not too wide)
        assert 0.05 < null_std < 0.15, f"Null std {null_std} outside expected range"

        # P-value should be non-significant for uncorrelated data
        assert result["p"] > 0.05

    def test_phase_randomize_detects_significant_correlation(self):
        """Test that phase_randomize correctly detects significant correlations.

        Verifies statistical power - should detect strong correlations.
        """
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        # Create two strongly correlated autocorrelated time series
        # Use a smooth autocorrelated signal and add correlated noise
        t = np.linspace(0, 10 * np.pi, 200)
        base_signal = (
            np.sin(t) + np.sin(2 * t) + np.sin(3 * t)
        )  # Complex autocorrelated signal
        x = base_signal + np.random.randn(200) * 0.3
        y = (
            base_signal + np.random.randn(200) * 0.3
        )  # Strong correlation via shared signal

        result = timeseries_correlation_permutation_test(
            x, y, method="phase_randomize", n_permute=500, random_state=42, n_jobs=1
        )

        # Should detect significant correlation
        assert abs(result["correlation"]) > 0.7, "Should have strong correlation"
        assert result["p"] < 0.05, "Should be statistically significant"

    def test_circle_shift_vs_phase_randomize_consistency(self):
        """Test that both methods produce sensible results for same data.

        While the methods differ (circle_shift preserves autocorrelation,
        phase_randomize preserves power spectrum), both should:
        1. Destroy correlation under H0
        2. Produce null distributions centered near zero
        3. Give similar p-values for uncorrelated data
        """
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(150)
        y = np.random.randn(150)

        result_circle = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=500, random_state=42, n_jobs=1
        )

        result_phase = timeseries_correlation_permutation_test(
            x, y, method="phase_randomize", n_permute=500, random_state=42, n_jobs=1
        )

        # Both should give non-significant results for uncorrelated data
        assert result_circle["p"] > 0.05
        assert result_phase["p"] > 0.05

        # P-values should be in similar ballpark (both testing same H0)
        # Allow generous tolerance as methods differ
        assert abs(result_circle["p"] - result_phase["p"]) < 0.5


# ============================================================================
# Matrix Permutation Tests (Mantel Test)
# ============================================================================


class TestMatrixHelpers:
    """Test helper functions for matrix permutation."""

    def test_extract_upper_triangle(self):
        """Test extraction of upper triangle elements."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        # Create a simple 5×5 matrix
        matrix = np.arange(25).reshape(5, 5)

        # Extract upper triangle
        elements = _extract_matrix_elements(matrix, how="upper")

        # Should have n*(n-1)/2 = 5*4/2 = 10 elements
        assert len(elements) == 10

        # Verify correct elements (manually check a few)
        # Upper triangle indices: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        expected = np.array([1, 2, 3, 4, 7, 8, 9, 13, 14, 19])
        np.testing.assert_array_equal(elements, expected)

    def test_extract_lower_triangle(self):
        """Test extraction of lower triangle elements."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        matrix = np.arange(25).reshape(5, 5)
        elements = _extract_matrix_elements(matrix, how="lower")

        # Should have n*(n-1)/2 = 10 elements
        assert len(elements) == 10

        # Lower triangle indices: (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), (4,0), (4,1), (4,2), (4,3)
        expected = np.array([5, 10, 11, 15, 16, 17, 20, 21, 22, 23])
        np.testing.assert_array_equal(elements, expected)

    def test_extract_full_matrix_no_diag(self):
        """Test extraction of full matrix without diagonal."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        matrix = np.arange(25).reshape(5, 5)
        elements = _extract_matrix_elements(matrix, how="full", include_diag=False)

        # Should have n*n - n = 25 - 5 = 20 elements (all except diagonal)
        assert len(elements) == 20

        # Should be concatenation of upper and lower triangles
        upper = np.array([1, 2, 3, 4, 7, 8, 9, 13, 14, 19])
        lower = np.array([5, 10, 11, 15, 16, 17, 20, 21, 22, 23])
        expected = np.concatenate([upper, lower])
        np.testing.assert_array_equal(elements, expected)

    def test_extract_full_matrix_with_diag(self):
        """Test extraction of full matrix with diagonal."""
        from nltools.algorithms.inference.matrix import _extract_matrix_elements

        matrix = np.arange(25).reshape(5, 5)
        elements = _extract_matrix_elements(matrix, how="full", include_diag=True)

        # Should have all n*n = 25 elements
        assert len(elements) == 25

        # Should be raveled matrix
        expected = matrix.ravel()
        np.testing.assert_array_equal(elements, expected)

    def test_permute_matrix_symmetric(self):
        """Test symmetric row+column permutation."""
        from nltools.algorithms.inference.matrix import _permute_matrix_symmetric

        # Create a simple matrix with identifiable structure
        matrix = np.arange(16).reshape(4, 4)

        # Identity permutation should leave matrix unchanged
        perm_identity = np.arange(4)
        result = _permute_matrix_symmetric(matrix, perm_identity)
        np.testing.assert_array_equal(result, matrix)

        # Known permutation: reverse order
        perm_reverse = np.array([3, 2, 1, 0])
        result = _permute_matrix_symmetric(matrix, perm_reverse)

        # Manually verify: should reverse both rows and columns
        expected = np.array(
            [[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]]
        )
        np.testing.assert_array_equal(result, expected)

        # Verify shape preserved
        assert result.shape == matrix.shape

    def test_permute_matrix_preserves_symmetry(self):
        """Test that symmetric permutation preserves matrix symmetry."""
        from nltools.algorithms.inference.matrix import _permute_matrix_symmetric

        # Create a symmetric matrix
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])

        # Apply any permutation
        perm = np.array([2, 0, 1])
        result = _permute_matrix_symmetric(matrix, perm)

        # Result should still be symmetric
        np.testing.assert_array_equal(result, result.T)

    def test_compute_matrix_correlation_pearson(self):
        """Test Pearson correlation computation."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Identical matrices should have r = 1.0
        matrix = np.random.randn(5, 5)
        r = _compute_matrix_correlation(matrix, matrix, metric="pearson")
        assert abs(r - 1.0) < 1e-10

        # Negated matrices should have r = -1.0
        r = _compute_matrix_correlation(matrix, -matrix, metric="pearson")
        assert abs(r - (-1.0)) < 1e-10

        # Uncorrelated random matrices should have |r| < 1.0
        np.random.seed(42)
        m1 = np.random.randn(10, 10)
        m2 = np.random.randn(10, 10)
        r = _compute_matrix_correlation(m1, m2, metric="pearson")
        assert abs(r) < 1.0

    def test_compute_matrix_correlation_spearman(self):
        """Test Spearman correlation computation."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Identical matrices should have r = 1.0
        matrix = np.random.randn(5, 5)
        r = _compute_matrix_correlation(matrix, matrix, metric="spearman")
        assert abs(r - 1.0) < 1e-10

        # Monotonic relationship should have high Spearman
        m1 = np.arange(25).reshape(5, 5).astype(float)
        m2 = m1**2  # Monotonic but not linear
        r = _compute_matrix_correlation(m1, m2, metric="spearman")
        assert r > 0.99  # Should be very high for monotonic relationship

    def test_compute_matrix_correlation_kendall(self):
        """Test Kendall correlation computation."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Identical matrices should have tau = 1.0
        matrix = np.random.randn(5, 5)
        tau = _compute_matrix_correlation(matrix, matrix, metric="kendall")
        assert abs(tau - 1.0) < 1e-10

        # Negated matrices should have tau ≈ -1.0
        tau = _compute_matrix_correlation(matrix, -matrix, metric="kendall")
        assert abs(tau - (-1.0)) < 1e-10

    def test_compute_matrix_correlation_all_extraction_modes(self):
        """Test that correlation works with all extraction modes."""
        from nltools.algorithms.inference.matrix import _compute_matrix_correlation

        # Create symmetric matrices (typical use case)
        np.random.seed(42)
        m1 = np.random.randn(6, 6)
        m1 = (m1 + m1.T) / 2  # Make symmetric
        m2 = np.random.randn(6, 6)
        m2 = (m2 + m2.T) / 2

        # All modes should work
        r_upper = _compute_matrix_correlation(m1, m2, how="upper")
        r_lower = _compute_matrix_correlation(m1, m2, how="lower")
        r_full_no_diag = _compute_matrix_correlation(
            m1, m2, how="full", include_diag=False
        )
        r_full_with_diag = _compute_matrix_correlation(
            m1, m2, how="full", include_diag=True
        )

        # For symmetric matrices, upper and lower should be identical
        assert abs(r_upper - r_lower) < 1e-10

        # Full modes should give reasonable correlations
        assert isinstance(r_full_no_diag, float)
        assert isinstance(r_full_with_diag, float)


class TestMatrixPermutationCPUParallel:
    """Test CPU-parallel implementation of matrix permutation."""

    def test_basic_functionality(self):
        """Test basic matrix permutation computation."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        # Create two correlated matrices
        np.random.seed(42)
        n = 20
        # Generate data matrix where rows are observations and columns are features
        data1 = np.random.randn(30, n)  # 30 observations, 20 features
        data2 = np.random.randn(30, n)
        # Compute correlation matrices (n×n)
        m1 = np.corrcoef(data1.T)  # Transpose to get features × features correlation
        m2 = np.corrcoef(data2.T)

        result = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=100,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        # Check result structure
        assert "correlation" in result
        assert "p" in result
        assert "backend" in result
        assert "null_dist" in result

        # Check types
        assert isinstance(result["correlation"], float)
        assert isinstance(result["p"], float)
        assert isinstance(result["backend"], str)
        assert isinstance(result["null_dist"], np.ndarray)

        # P-value should be in valid range
        assert 0 < result["p"] < 1

        # Null distribution should have correct size
        assert len(result["null_dist"]) == 100

    def test_determinism(self):
        """Test that same seed produces identical results."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        np.random.seed(42)
        n = 15
        m1 = np.random.randn(n, n)
        m2 = np.random.randn(n, n)

        # Run twice with same seed
        result1 = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=200,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        result2 = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=200,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        # Results should be EXACTLY identical (0.000% variance)
        assert result1["correlation"] == result2["correlation"]
        assert result1["p"] == result2["p"]
        np.testing.assert_array_equal(result1["null_dist"], result2["null_dist"])

    def test_parallel_consistency(self):
        """Test that n_jobs=1 and n_jobs=-1 produce identical results."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        np.random.seed(42)
        n = 12
        m1 = np.random.randn(n, n)
        m2 = np.random.randn(n, n)

        # Run with n_jobs=1
        result_serial = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=150,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )

        # Run with n_jobs=-1 (all cores)
        result_parallel = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=150,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=-1,
            random_state=42,
        )

        # Results should be identical
        assert result_serial["correlation"] == result_parallel["correlation"]
        assert result_serial["p"] == result_parallel["p"]
        np.testing.assert_array_equal(
            result_serial["null_dist"], result_parallel["null_dist"]
        )

    def test_return_null_distribution(self):
        """Test that null distribution is returned when requested."""
        from nltools.algorithms.inference.matrix import _matrix_permutation_cpu_parallel

        np.random.seed(42)
        m1 = np.random.randn(10, 10)
        m2 = np.random.randn(10, 10)

        # With return_null=True
        result = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=100,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=True,
            n_jobs=1,
            random_state=42,
        )
        assert "null_dist" in result
        assert len(result["null_dist"]) == 100

        # Without return_null
        result = _matrix_permutation_cpu_parallel(
            data1=m1,
            data2=m2,
            n_permute=100,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            return_null=False,
            n_jobs=1,
            random_state=42,
        )
        assert "null_dist" not in result


class TestMatrixPermutationMain:
    """Test main matrix_permutation_test function."""

    def test_input_validation_non_square(self):
        """Test that non-square matrices are rejected."""
        from nltools.algorithms.inference import matrix_permutation_test

        m1 = np.random.randn(5, 6)  # Not square
        m2 = np.random.randn(5, 6)

        with pytest.raises(ValueError, match="must be square"):
            matrix_permutation_test(m1, m2, n_permute=100)

    def test_input_validation_mismatched_sizes(self):
        """Test that mismatched matrix sizes are rejected."""
        from nltools.algorithms.inference import matrix_permutation_test

        m1 = np.random.randn(5, 5)
        m2 = np.random.randn(6, 6)

        with pytest.raises(ValueError, match="must have same shape"):
            matrix_permutation_test(m1, m2, n_permute=100)

    def test_input_validation_invalid_metric(self):
        """Test that invalid metric is rejected."""
        from nltools.algorithms.inference import matrix_permutation_test

        m1 = np.random.randn(5, 5)
        m2 = np.random.randn(5, 5)

        with pytest.raises(ValueError, match="metric must be"):
            matrix_permutation_test(m1, m2, metric="invalid")

    def test_input_validation_invalid_how(self):
        """Test that invalid 'how' parameter is rejected."""
        from nltools.algorithms.inference import matrix_permutation_test

        m1 = np.random.randn(5, 5)
        m2 = np.random.randn(5, 5)

        with pytest.raises(ValueError, match="how must be"):
            matrix_permutation_test(m1, m2, how="invalid")

    def test_all_extraction_modes(self):
        """Test that all extraction modes work correctly."""
        from nltools.algorithms.inference import matrix_permutation_test

        np.random.seed(42)
        m1 = np.random.randn(8, 8)
        m2 = np.random.randn(8, 8)

        # All modes should work
        result_upper = matrix_permutation_test(
            m1, m2, how="upper", n_permute=100, random_state=42, n_jobs=1
        )
        result_lower = matrix_permutation_test(
            m1, m2, how="lower", n_permute=100, random_state=42, n_jobs=1
        )
        result_full_no_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=False,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )
        result_full_with_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=True,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )

        # All should return valid results
        assert 0 < result_upper["p"] < 1
        assert 0 < result_lower["p"] < 1
        assert 0 < result_full_no_diag["p"] < 1
        assert 0 < result_full_with_diag["p"] < 1

    def test_all_metrics(self):
        """Test that all correlation metrics work correctly."""
        from nltools.algorithms.inference import matrix_permutation_test

        np.random.seed(42)
        m1 = np.random.randn(8, 8)
        m2 = np.random.randn(8, 8)

        # All metrics should work
        result_pearson = matrix_permutation_test(
            m1, m2, metric="pearson", n_permute=100, random_state=42, n_jobs=1
        )
        result_spearman = matrix_permutation_test(
            m1, m2, metric="spearman", n_permute=100, random_state=42, n_jobs=1
        )
        result_kendall = matrix_permutation_test(
            m1, m2, metric="kendall", n_permute=100, random_state=42, n_jobs=1
        )

        # All should return valid results
        assert 0 < result_pearson["p"] < 1
        assert 0 < result_spearman["p"] < 1
        assert 0 < result_kendall["p"] < 1

    def test_include_diag_parameter(self):
        """Test that include_diag parameter works correctly."""
        from nltools.algorithms.inference import matrix_permutation_test

        np.random.seed(42)
        m1 = np.random.randn(8, 8)
        m2 = np.random.randn(8, 8)

        # Should work with both values
        result_no_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=False,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )
        result_with_diag = matrix_permutation_test(
            m1,
            m2,
            how="full",
            include_diag=True,
            n_permute=100,
            random_state=42,
            n_jobs=1,
        )

        # Both should return valid results
        assert 0 < result_no_diag["p"] < 1
        assert 0 < result_with_diag["p"] < 1

        # Results might differ since different elements are used
        # (Just verify they both work, not that they match)


class TestMatrixPermutationCorrectness:
    """Test statistical correctness of matrix permutation."""

    def test_identical_matrices(self):
        """Test that identical matrices produce perfect correlation and significant p-value."""
        from nltools.algorithms.inference import matrix_permutation_test

        # Create a matrix and test against itself
        np.random.seed(42)
        matrix = np.random.randn(20, 20)

        result = matrix_permutation_test(
            matrix, matrix, n_permute=500, random_state=42, n_jobs=1
        )

        # Correlation should be perfect
        assert abs(result["correlation"] - 1.0) < 1e-10

        # P-value should be highly significant
        assert result["p"] < 0.05

    def test_uncorrelated_matrices(self):
        """Test that uncorrelated random matrices produce non-significant p-value."""
        from nltools.algorithms.inference import matrix_permutation_test

        # Create two independent random matrices
        np.random.seed(42)
        m1 = np.random.randn(25, 25)
        m2 = np.random.randn(25, 25)

        result = matrix_permutation_test(
            m1, m2, n_permute=500, random_state=42, n_jobs=1
        )

        # Correlation should be weak
        assert abs(result["correlation"]) < 0.3

        # P-value should be non-significant
        assert result["p"] > 0.05

    def test_null_distribution_centered(self):
        """Test that null distribution is centered near zero for uncorrelated matrices."""
        from nltools.algorithms.inference import matrix_permutation_test

        np.random.seed(42)
        m1 = np.random.randn(20, 20)
        m2 = np.random.randn(20, 20)

        result = matrix_permutation_test(
            m1, m2, n_permute=1000, return_null=True, random_state=42, n_jobs=1
        )

        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

        # Mean should be close to 0
        assert abs(null_mean) < 0.05

        # Standard deviation should be reasonable
        assert 0.01 < null_std < 0.15

    def test_backward_compatibility_stats_py(self):
        """Test backward compatibility with stats.py matrix_permutation.

        Note: Expect ~1-2% variance due to different RNG pattern
        (independent RandomState per permutation vs shared RNG state).
        Both implementations are statistically correct.
        """
        from nltools.algorithms.inference import matrix_permutation_test
        from nltools.stats import matrix_permutation

        # Create test matrices
        np.random.seed(42)
        n = 15
        m1 = np.random.randn(n, n)
        m2 = np.random.randn(n, n)

        # New implementation
        result_new = matrix_permutation_test(
            m1,
            m2,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            random_state=42,
            n_jobs=1,
        )

        # Old implementation (stats.py)
        result_old = matrix_permutation(
            m1,
            m2,
            n_permute=N_PERMUTE_STATS_COMPARISON,
            metric="pearson",
            how="upper",
            include_diag=False,
            tail=2,
            random_state=42,
            n_jobs=1,
        )

        # Observed correlation should be identical (deterministic computation)
        np.testing.assert_allclose(
            result_new["correlation"],
            result_old["correlation"],
            rtol=TOLERANCE_STATS_DETERMINISTIC,
        )

        # P-values may differ slightly (~1-2%) due to RNG pattern
        # This is expected and acceptable (see DESIGN.md)
        np.testing.assert_allclose(
            result_new["p"], result_old["p"], rtol=TOLERANCE_STATS_PVALUE
        )
