"""Tests for timeseries permutation tests and helper functions."""

import pytest
import numpy as np

from nltools.algorithms.inference.timeseries import (
    circle_shift,
    phase_randomize,
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
# GPU Timeseries Tests
# ============================================================================


class TestTimeseriesGPU:
    """Tests for GPU-accelerated timeseries permutation tests."""

    @pytest.mark.tier1
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
            x, y, method="circle_shift", n_permute=100, backend="torch", random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert "backend" in result
        assert "torch" in result["backend"]
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.tier1
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
            backend="torch",
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result
        assert "backend" in result
        assert "torch" in result["backend"]
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.tier1
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
            x, y, method="circle_shift", n_permute=100, backend="torch", random_state=42
        )
        result2 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, backend="torch", random_state=42
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
            backend="torch",
            return_null=True,
            random_state=42,
        )

        assert "null_distribution" in result
        assert result["null_distribution"].shape == (100,)

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
            x, y, method="circle_shift", n_permute=100, backend=None, random_state=42
        )
        result_gpu = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, backend="torch", random_state=42
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
            backend=None,
            random_state=42,
        )
        result_gpu = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=100,
            backend="torch",
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
            backend="torch",
            max_gpu_memory_gb=1.0,  # Small memory budget to force batching
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result
        assert result["null_distribution"].shape == (5000,)
