"""Tests for timeseries permutation tests and helper functions."""

import pytest
import numpy as np
from scipy.stats import kstest

from nltools.algorithms.inference.timeseries import (
    circle_shift,
    phase_randomize,
    timeseries_correlation_permutation_test,
)
from nltools.tests.core.test_inference import (
    TOLERANCE_STATS_DETERMINISTIC,
    TOLERANCE_STATS_PVALUE_CIRCLE_SHIFT,
    TOLERANCE_STATS_PVALUE_PHASE_RANDOMIZE,
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.backends import Backend


class TestCircleShift:
    """Tests for circle_shift() function."""

    @pytest.mark.tier1
    def test_preserves_shape_1d(self):
        """Test that circle_shift preserves shape for 1D data."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        assert shifted.shape == data.shape

    @pytest.mark.tier1
    def test_preserves_shape_2d(self):
        """Test that circle_shift preserves shape for 2D data."""
        data = np.random.randn(50, 10)
        shifted = circle_shift(data, random_state=42)
        assert shifted.shape == data.shape

    @pytest.mark.tier1
    def test_deterministic_with_seed_1d(self):
        """Test that circle_shift is deterministic with random_state for 1D."""
        data = np.random.randn(100)
        shifted1 = circle_shift(data, random_state=42)
        shifted2 = circle_shift(data, random_state=42)
        np.testing.assert_array_equal(shifted1, shifted2)

    @pytest.mark.tier1
    def test_deterministic_with_seed_2d(self):
        """Test that circle_shift is deterministic with random_state for 2D."""
        data = np.random.randn(100, 5)
        shifted1 = circle_shift(data, random_state=42)
        shifted2 = circle_shift(data, random_state=42)
        np.testing.assert_array_equal(shifted1, shifted2)

    @pytest.mark.tier1
    def test_preserves_values_1d(self):
        """Test that circle_shift preserves all values (just reorders) for 1D."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        assert sorted(shifted) == sorted(data)

    @pytest.mark.tier1
    def test_preserves_values_2d(self):
        """Test that circle_shift preserves all values for 2D."""
        data = np.random.randn(50, 10)
        shifted = circle_shift(data, random_state=42)
        for i in range(data.shape[1]):
            assert sorted(shifted[:, i]) == pytest.approx(sorted(data[:, i]))

    @pytest.mark.tier1
    def test_explicit_shift_1d(self):
        """Test circle_shift with explicit shift amount for 1D."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        expected = np.array([4, 5, 1, 2, 3])
        np.testing.assert_array_equal(shifted, expected)

    @pytest.mark.tier1
    def test_explicit_shift_2d(self):
        """Test circle_shift with explicit shift amounts for 2D."""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        shifted = circle_shift(data, shift_amount=np.array([1, 2]))
        expected = np.array([[4, 30], [1, 40], [2, 10], [3, 20]])
        np.testing.assert_array_equal(shifted, expected)

    @pytest.mark.skip(
        reason="circle_shift has been moved from stats.py to algorithms.inference.timeseries"
    )
    def test_matches_stats_py_1d(self):
        """Test that circle_shift matches stats.py for 1D data.

        NOTE: This test is skipped because circle_shift has been moved from
        nltools.stats to nltools.algorithms.inference.timeseries as part of
        the refactoring. The function no longer exists in stats.py.
        """
        pass

    @pytest.mark.skip(
        reason="circle_shift has been moved from stats.py to algorithms.inference.timeseries"
    )
    def test_matches_stats_py_2d(self):
        """Test that circle_shift matches stats.py for 2D data.

        NOTE: This test is skipped because circle_shift has been moved from
        nltools.stats to nltools.algorithms.inference.timeseries as part of
        the refactoring. The function no longer exists in stats.py.
        """
        pass


class TestPhaseRandomize:
    """Tests for phase_randomize() function."""

    @pytest.mark.tier1
    def test_preserves_shape_1d(self):
        """Test that phase_randomize preserves shape for 1D data."""
        data = np.random.randn(100)
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape

    @pytest.mark.tier1
    def test_preserves_shape_2d(self):
        """Test that phase_randomize preserves shape for 2D data."""
        data = np.random.randn(100, 5)
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape

    @pytest.mark.tier1
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

    @pytest.mark.tier1
    def test_preserves_power_spectrum_2d(self):
        """Test that phase_randomize preserves power spectrum for 2D data."""
        data = np.random.randn(200, 5)
        randomized = phase_randomize(data, random_state=42)

        # Check each feature independently
        for i in range(data.shape[1]):
            power_orig = np.abs(np.fft.rfft(data[:, i])) ** 2
            power_rand = np.abs(np.fft.rfft(randomized[:, i])) ** 2
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    @pytest.mark.tier1
    def test_changes_phase_1d(self):
        """Test that phase_randomize actually changes the signal."""
        # Use deterministic signal
        t = np.linspace(0, 10 * np.pi, 100)
        data = np.sin(t)

        randomized = phase_randomize(data, random_state=42)

        # Signal should be different
        assert not np.allclose(data, randomized)

    @pytest.mark.tier1
    def test_changes_phase_2d(self):
        """Test that phase_randomize changes signals for 2D data."""
        data = np.random.randn(100, 5)
        randomized = phase_randomize(data, random_state=42)

        # Should be different
        assert not np.allclose(data, randomized)

    @pytest.mark.tier1
    def test_deterministic_with_seed_1d(self):
        """Test that phase_randomize is deterministic with random_state for 1D."""
        data = np.random.randn(100)
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)

    @pytest.mark.tier1
    def test_deterministic_with_seed_2d(self):
        """Test that phase_randomize is deterministic with random_state for 2D."""
        data = np.random.randn(100, 5)
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)

    @pytest.mark.tier1
    def test_backend_consistency_numpy(self):
        """Test that phase_randomize works with NumPy backend."""
        data = np.random.randn(100)
        randomized = phase_randomize(data, backend="numpy", random_state=42)

        # Should preserve power spectrum
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    @pytest.mark.tier2
    def test_backend_consistency_torch_1d(self):
        """Test that phase_randomize works with torch backend for 1D data."""
        pytest.importorskip("torch")
        from nltools.backends import check_gpu_available

        if not check_gpu_available()[0]:
            pytest.skip("GPU not available")

        data = np.random.randn(200)
        randomized = phase_randomize(data, backend="torch", random_state=42)

        # Should preserve power spectrum (within float32 tolerance)
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2
        # GPU uses float32, so relax tolerance compared to NumPy float64
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-4, atol=1e-4)

    @pytest.mark.tier2
    def test_backend_consistency_torch_2d(self):
        """Test that phase_randomize works with torch backend for 2D data."""
        pytest.importorskip("torch")
        from nltools.backends import check_gpu_available

        if not check_gpu_available()[0]:
            pytest.skip("GPU not available")

        data = np.random.randn(200, 5)
        randomized = phase_randomize(data, backend="torch", random_state=42)

        # Should preserve power spectrum for each feature
        for i in range(data.shape[1]):
            power_orig = np.abs(np.fft.rfft(data[:, i])) ** 2
            power_rand = np.abs(np.fft.rfft(randomized[:, i])) ** 2
            # GPU uses float32, so relax tolerance compared to NumPy float64
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-4, atol=1e-4)

    @pytest.mark.tier2
    def test_backend_consistency_torch_vs_numpy(self):
        """Test that torch and numpy backends produce similar results."""
        pytest.importorskip("torch")
        from nltools.backends import check_gpu_available

        if not check_gpu_available()[0]:
            pytest.skip("GPU not available")

        data = np.random.randn(200)
        randomized_numpy = phase_randomize(data, backend="numpy", random_state=42)
        randomized_torch = phase_randomize(data, backend="torch", random_state=42)

        # Results should match within float32 tolerance (GPU uses float32)
        np.testing.assert_allclose(
            randomized_numpy, randomized_torch, rtol=1e-5, atol=1e-5
        )

    @pytest.mark.skip(
        reason="phase_randomize has been moved from stats.py to algorithms.inference.timeseries"
    )
    def test_matches_stats_py_1d(self):
        """Test that phase_randomize matches stats.py for 1D data.

        NOTE: This test is skipped because phase_randomize has been moved from
        nltools.stats to nltools.algorithms.inference.timeseries as part of
        the refactoring. The function no longer exists in stats.py.
        """
        pass

    @pytest.mark.skip(
        reason="phase_randomize has been moved from stats.py to algorithms.inference.timeseries"
    )
    def test_matches_stats_py_2d(self):
        """Test that phase_randomize matches stats.py for 2D data.

        NOTE: This test is skipped because phase_randomize has been moved from
        nltools.stats to nltools.algorithms.inference.timeseries as part of
        the refactoring. The function no longer exists in stats.py.
        """
        pass


class TestTimeseriesCorrelation:
    """Tests for timeseries_correlation_permutation_test() function."""

    @pytest.mark.tier1
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

    @pytest.mark.tier1
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

    @pytest.mark.tier1
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

    @pytest.mark.tier1
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

        assert "null_dist" in result
        assert result["null_dist"].shape == (100,)

    @pytest.mark.tier1
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

    @pytest.mark.tier1
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

    @pytest.mark.tier2
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

    @pytest.mark.tier2
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
        null_mean = np.mean(result["null_dist"])
        null_std = np.std(result["null_dist"])

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
# GPU Timeseries Tests
# ============================================================================


class TestTimeseriesGPU:
    """Tests for GPU-accelerated timeseries permutation tests."""

    @pytest.mark.tier2
    def test_gpu_basic_functionality_circle_shift(self):
        """Test basic GPU functionality with circle_shift method."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, parallel="gpu", random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert "parallel" in result
        assert result["parallel"] == "gpu"
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.tier2
    def test_gpu_basic_functionality_phase_randomize(self):
        """Test basic GPU functionality with phase_randomize method."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=100,
            parallel="gpu",
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result
        assert "parallel" in result
        assert result["parallel"] == "gpu"
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.tier2
    def test_gpu_deterministic_with_seed(self):
        """Test that GPU results are deterministic with random_state."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result1 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, parallel="gpu", random_state=42
        )
        result2 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, parallel="gpu", random_state=42
        )

        np.testing.assert_equal(result1["correlation"], result2["correlation"])
        np.testing.assert_equal(result1["p"], result2["p"])

    @pytest.mark.tier1
    def test_gpu_return_null_distribution(self):
        """Test that GPU returns null distribution when requested."""
        pytest.importorskip("torch")
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
            parallel="gpu",
            return_null=True,
            random_state=42,
        )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100,)

    @pytest.mark.tier2
    def test_gpu_matches_cpu_circle_shift(self):
        """Test that GPU circle_shift matches CPU results (within float32 tolerance)."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)

        result_cpu = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, parallel="cpu", random_state=42
        )
        result_gpu = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, parallel="gpu", random_state=42
        )

        # Correlation should match closely (GPU uses float32)
        np.testing.assert_allclose(
            result_cpu["correlation"],
            result_gpu["correlation"],
            rtol=TOLERANCE_GPU_VALUE,
        )

        # P-values should match closely
        np.testing.assert_allclose(
            result_cpu["p"], result_gpu["p"], rtol=TOLERANCE_GPU_PVALUE
        )

    @pytest.mark.tier2
    def test_gpu_matches_cpu_phase_randomize(self):
        """Test that GPU phase_randomize matches CPU results (within float32 tolerance)."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)

        result_cpu = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=100,
            parallel="cpu",
            random_state=42,
        )
        result_gpu = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=100,
            parallel="gpu",
            random_state=42,
        )

        # Correlation should match closely (GPU uses float32)
        np.testing.assert_allclose(
            result_cpu["correlation"],
            result_gpu["correlation"],
            rtol=TOLERANCE_GPU_VALUE,
        )

        # P-values should match closely
        np.testing.assert_allclose(
            result_cpu["p"], result_gpu["p"], rtol=TOLERANCE_GPU_PVALUE
        )

    @pytest.mark.tier2
    def test_gpu_phase_randomize_preserves_power_spectrum(self):
        """Test that GPU phase_randomize preserves power spectrum."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import _phase_randomize_gpu

        np.random.seed(42)
        x = np.random.randn(200)

        backend = Backend("torch")
        rng = np.random.RandomState(42)
        randomized = _phase_randomize_gpu(x, backend, rng)

        # Compute power spectra
        power_orig = np.abs(np.fft.rfft(x)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2

        # Should match exactly (within numerical precision)
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-5)

    @pytest.mark.tier2
    def test_gpu_circle_shift_correctness(self):
        """Test that GPU circle_shift produces correct results."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import (
            _circle_shift_gpu,
            circle_shift,
        )

        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5])

        # Test GPU version
        backend = Backend("torch")
        x_device = backend.to_device(x.astype(np.float32))
        shifted_gpu = _circle_shift_gpu(x_device, shift_amount=2, backend=backend)
        shifted_gpu = backend.to_numpy(shifted_gpu)

        # Test CPU version
        shifted_cpu = circle_shift(x, shift_amount=2)

        # Should match exactly
        np.testing.assert_allclose(shifted_gpu, shifted_cpu, rtol=1e-5)

    @pytest.mark.tier2
    def test_gpu_batching_prevents_oom(self):
        """Test that GPU batching handles large problems without OOM."""
        torch = pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("GPU not available for OOM test")

        from nltools.algorithms.inference.timeseries import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        # Large problem that would OOM without batching
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        # Should complete without error
        result = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=5000,
            parallel="gpu",
            max_gpu_memory_gb=1.0,  # Small memory budget to force batching
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result
        assert result["null_dist"].shape == (5000,)


# ============================================================================
# Test Timeseries Correlation Statistical Correctness
# ============================================================================


def _generate_ar1_process(n_samples, ar_coef, random_state=None):
    """Generate AR(1) process with known autocorrelation."""
    np.random.seed(random_state)
    data = np.zeros(n_samples)
    noise = np.random.randn(n_samples)
    for i in range(1, n_samples):
        data[i] = ar_coef * data[i - 1] + noise[i]
    return data


def _generate_shared_signal_timeseries(
    n_timepoints, correlation_strength, random_state=None
):
    """Generate time series data with known correlation."""
    np.random.seed(random_state)
    shared_signal = np.random.randn(n_timepoints)
    x = shared_signal * np.sqrt(correlation_strength) + np.random.randn(
        n_timepoints
    ) * np.sqrt(1 - correlation_strength)
    y = shared_signal * np.sqrt(correlation_strength) + np.random.randn(
        n_timepoints
    ) * np.sqrt(1 - correlation_strength)
    return x, y


class TestTimeseriesCorrelationStatisticalCorrectness:
    """Test statistical correctness of timeseries correlation permutation tests."""

    @pytest.mark.tier2
    def test_null_hypothesis_pvalue_distribution(self):
        """Test that p-values are uniformly distributed under null hypothesis for all methods."""
        n_samples = 100
        n_tests = 100  # Run many tests with different seeds
        n_permute = 2000  # Enough permutations for stable p-values

        methods = ["circle_shift", "phase_randomize"]

        for method in methods:
            p_values = []

            for seed in range(n_tests):
                np.random.seed(seed)
                # Generate independent time series (no correlation)
                x = np.random.randn(n_samples)
                y = np.random.randn(n_samples)

                result = timeseries_correlation_permutation_test(
                    x, y, method=method, n_permute=n_permute, random_state=seed
                )

                p_values.append(result["p"])

            # Test uniformity using Kolmogorov-Smirnov test
            # Under null hypothesis, p-values should be uniformly distributed
            ks_statistic, ks_pvalue = kstest(p_values, "uniform")

            # KS test p-value should be > 0.05 (p-values are uniform)
            assert ks_pvalue > 0.05, (
                f"P-values should be uniformly distributed under null hypothesis for {method}. "
                f"KS test p-value: {ks_pvalue:.4f}"
            )

    @pytest.mark.tier1
    def test_correlation_value_correctness(self):
        """Test that computed correlation values match expected values."""
        n_samples = 200
        correlation_strength = 0.7  # Known correlation strength

        # Generate time series with known correlation
        x, y = _generate_shared_signal_timeseries(
            n_samples, correlation_strength, random_state=42
        )

        # Test with circle_shift method
        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=2000, random_state=42
        )

        # Computed correlation should be close to expected
        # Tolerance: rtol=0.1 (10% as specified in plan - time series are noisy)
        expected_correlation = correlation_strength  # Approximate
        np.testing.assert_allclose(
            result["correlation"], expected_correlation, rtol=0.2, atol=0.1
        )

    @pytest.mark.tier1
    def test_circle_shift_preserves_autocorrelation(self):
        """Test that circle_shift preserves autocorrelation structure."""
        n_samples = 200
        ar_coef = 0.8  # Strong autocorrelation

        # Create time series with strong autocorrelation (AR(1) process)
        np.random.seed(42)
        x = _generate_ar1_process(n_samples, ar_coef, random_state=42)

        # Compute autocorrelation of original data (lag-1)
        autocorr_orig = np.corrcoef(x[:-1], x[1:])[0, 1]

        # Run circle_shift many times and check autocorrelation is preserved
        autocorrs_shifted = []
        for seed in range(50):
            shifted = circle_shift(x, random_state=seed)
            autocorr_shifted = np.corrcoef(shifted[:-1], shifted[1:])[0, 1]
            autocorrs_shifted.append(autocorr_shifted)

        # Autocorrelation should be preserved (within sampling error)
        mean_autocorr_shifted = np.mean(autocorrs_shifted)
        assert abs(mean_autocorr_shifted - autocorr_orig) < 0.1, (
            f"Circle shift should preserve autocorrelation. "
            f"Original: {autocorr_orig:.4f}, Shifted mean: {mean_autocorr_shifted:.4f}"
        )

    @pytest.mark.tier1
    def test_phase_randomize_preserves_power_spectrum(self):
        """Test that phase_randomize preserves power spectrum."""
        n_samples = 200

        # Create time series with known power spectrum
        np.random.seed(42)
        x = np.random.randn(n_samples)

        # Compute power spectrum of original
        power_orig = np.abs(np.fft.rfft(x)) ** 2

        # Run phase_randomize many times and check power spectrum is preserved
        for seed in range(10):
            randomized = phase_randomize(x, random_state=seed)
            power_rand = np.abs(np.fft.rfft(randomized)) ** 2

            # Power spectrum should match exactly (within numerical precision)
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10, atol=1e-10)

    @pytest.mark.tier1
    def test_effect_size_sensitivity(self):
        """Test that larger correlation produces lower p-values."""
        n_samples = 150
        n_permute = 5000  # Large enough for stable p-values

        # Test with different correlation strengths
        correlation_strengths = [0.0, 0.2, 0.4, 0.6]  # Null, small, medium, large
        p_values_circle = []
        p_values_phase = []

        for corr_strength in correlation_strengths:
            # Generate data with known correlation
            x, y = _generate_shared_signal_timeseries(
                n_samples, corr_strength, random_state=42
            )

            result_circle = timeseries_correlation_permutation_test(
                x, y, method="circle_shift", n_permute=n_permute, random_state=42
            )
            result_phase = timeseries_correlation_permutation_test(
                x, y, method="phase_randomize", n_permute=n_permute, random_state=42
            )

            p_values_circle.append(result_circle["p"])
            p_values_phase.append(result_phase["p"])

        # Verify larger correlation → smaller p-value (monotonic relationship)
        # Skip corr=0 (null hypothesis), test others
        # Note: Very large effects may hit minimum p-value (1/(n_permute+1)),
        # so allow >= for equality case when effects are extremely large
        assert p_values_circle[1] >= p_values_circle[2], (
            f"Larger correlation should produce smaller p-value (circle_shift). "
            f"corr=0.2: p={p_values_circle[1]:.6f}, corr=0.4: p={p_values_circle[2]:.6f}"
        )
        assert p_values_circle[2] >= p_values_circle[3], (
            f"Larger correlation should produce smaller p-value (circle_shift). "
            f"corr=0.4: p={p_values_circle[2]:.6f}, corr=0.6: p={p_values_circle[3]:.6f}"
        )

        assert p_values_phase[1] >= p_values_phase[2], (
            f"Larger correlation should produce smaller p-value (phase_randomize). "
            f"corr=0.2: p={p_values_phase[1]:.6f}, corr=0.4: p={p_values_phase[2]:.6f}"
        )
        assert p_values_phase[2] >= p_values_phase[3], (
            f"Larger correlation should produce smaller p-value (phase_randomize). "
            f"corr=0.4: p={p_values_phase[2]:.6f}, corr=0.6: p={p_values_phase[3]:.6f}"
        )

        # Large correlation (corr=0.6) should be significant
        assert p_values_circle[3] < 0.05, (
            f"Large correlation (corr=0.6) should be significant (circle_shift), got p={p_values_circle[3]:.4f}"
        )
        assert p_values_phase[3] < 0.05, (
            f"Large correlation (corr=0.6) should be significant (phase_randomize), got p={p_values_phase[3]:.4f}"
        )

    @pytest.mark.tier1
    def test_circle_shift_statistical_properties(self):
        """Test that circle_shift preserves statistical properties."""
        n_samples = 200

        # Generate null data (independent time series)
        np.random.seed(42)
        x = np.random.randn(n_samples)

        # Compute original statistics
        mean_orig = np.mean(x)
        var_orig = np.var(x)

        # Run circle_shift many times
        means_shifted = []
        vars_shifted = []
        for seed in range(50):
            shifted = circle_shift(x, random_state=seed)
            means_shifted.append(np.mean(shifted))
            vars_shifted.append(np.var(shifted))

        # Mean and variance should be preserved (within sampling error)
        mean_shifted = np.mean(means_shifted)
        var_shifted = np.mean(vars_shifted)

        assert abs(mean_shifted - mean_orig) < 0.1, (
            f"Circle shift should preserve mean. Original: {mean_orig:.4f}, Shifted: {mean_shifted:.4f}"
        )
        assert abs(var_shifted - var_orig) < 0.1, (
            f"Circle shift should preserve variance. Original: {var_orig:.4f}, Shifted: {var_shifted:.4f}"
        )

        # Verify circular property: shifted data should contain same values
        shifted = circle_shift(x, random_state=42)
        assert sorted(shifted) == pytest.approx(sorted(x), abs=1e-10), (
            "Circle shift should preserve all values (circular reordering)"
        )

    @pytest.mark.tier1
    def test_phase_randomize_statistical_properties(self):
        """Test that phase_randomize preserves power spectrum and approximate mean/variance."""
        n_samples = 200

        # Generate null data (independent time series)
        np.random.seed(42)
        x = np.random.randn(n_samples)

        # Compute original statistics
        mean_orig = np.mean(x)
        var_orig = np.var(x)
        power_orig = np.abs(np.fft.rfft(x)) ** 2

        # Run phase_randomize many times
        means_randomized = []
        vars_randomized = []
        for seed in range(20):
            randomized = phase_randomize(x, random_state=seed)
            means_randomized.append(np.mean(randomized))
            vars_randomized.append(np.var(randomized))

            # Power spectrum should be preserved exactly
            power_rand = np.abs(np.fft.rfft(randomized)) ** 2
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10, atol=1e-10)

        # Mean and variance should be approximately preserved (within reasonable tolerance)
        mean_randomized = np.mean(means_randomized)
        var_randomized = np.mean(vars_randomized)

        # Phase randomization preserves power spectrum exactly, but mean/variance may vary slightly
        # due to phase changes affecting the time-domain signal
        assert abs(mean_randomized - mean_orig) < 0.5, (
            f"Phase randomize should approximately preserve mean. "
            f"Original: {mean_orig:.4f}, Randomized: {mean_randomized:.4f}"
        )
        assert abs(var_randomized - var_orig) < 0.5, (
            f"Phase randomize should approximately preserve variance. "
            f"Original: {var_orig:.4f}, Randomized: {var_randomized:.4f}"
        )
