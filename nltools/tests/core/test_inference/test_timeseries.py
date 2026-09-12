"""Tests for timeseries permutation tests and helper functions."""

import pytest
import numpy as np

from nltools.algorithms import circle_shift, phase_randomize
from nltools.tests.core.test_inference import (
    TOLERANCE_GPU_VALUE,
    TOLERANCE_GPU_PVALUE,
)
from nltools.algorithms.backends import Backend, check_gpu_available


class TestCircleShift:
    """Tests for circle_shift() function."""

    def test_preserves_shape_1d(self):
        """Test that circle_shift preserves shape for 1D data."""
        data = np.array([1, 2, 3, 4, 5])
        shifted = circle_shift(data, shift_amount=2)
        assert shifted.shape == data.shape

    def test_preserves_shape_2d(self):
        """Test that circle_shift preserves shape for 2D data."""
        data = np.random.randn(30, 5)  # Reduced from 50, 10 for tier1 speed
        shifted = circle_shift(data, random_state=42)
        assert shifted.shape == data.shape

    def test_deterministic_with_seed_1d(self):
        """Test that circle_shift is deterministic with random_state for 1D."""
        data = np.random.randn(50)  # Reduced from 100 for tier1 speed
        shifted1 = circle_shift(data, random_state=42)
        shifted2 = circle_shift(data, random_state=42)
        np.testing.assert_array_equal(shifted1, shifted2)

    def test_random_shift_never_identity_1d(self):
        """A random 1D shift must never be the identity (shift of 0) (F016).

        The 1D path must draw from [1, len) like the 2D path, so no random
        seed should leave the data unshifted.
        """
        data = np.arange(10)
        for seed in range(200):
            shifted = circle_shift(data, random_state=seed)
            assert not np.array_equal(shifted, data), f"seed {seed} gave identity shift"

    def test_deterministic_with_seed_2d(self):
        """Test that circle_shift is deterministic with random_state for 2D."""
        data = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
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
        data = np.random.randn(30, 5)  # Reduced from 50, 10 for tier1 speed
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

    def test_preserves_autocorrelation(self):
        """Circle shift preserves the lag-1 autocorrelation that makes the null valid."""
        n_samples = 100
        ar_coef = 0.8

        # AR(1) process with strong autocorrelation
        np.random.seed(42)
        x = np.zeros(n_samples)
        noise = np.random.randn(n_samples)
        for i in range(1, n_samples):
            x[i] = ar_coef * x[i - 1] + noise[i]

        autocorr_orig = np.corrcoef(x[:-1], x[1:])[0, 1]

        autocorrs_shifted = []
        for seed in range(30):
            shifted = circle_shift(x, random_state=seed)
            autocorrs_shifted.append(np.corrcoef(shifted[:-1], shifted[1:])[0, 1])

        mean_autocorr_shifted = np.mean(autocorrs_shifted)
        assert abs(mean_autocorr_shifted - autocorr_orig) < 0.1, (
            f"Circle shift should preserve autocorrelation. "
            f"Original: {autocorr_orig:.4f}, Shifted mean: {mean_autocorr_shifted:.4f}"
        )


class TestPhaseRandomize:
    """Tests for phase_randomize() function."""

    def test_preserves_shape_1d(self):
        """Test that phase_randomize preserves shape for 1D data."""
        data = np.random.randn(50)  # Reduced from 100 for tier1 speed
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape

    def test_preserves_shape_2d(self):
        """Test that phase_randomize preserves shape for 2D data."""
        data = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
        randomized = phase_randomize(data, random_state=42)
        assert randomized.shape == data.shape

    def test_preserves_power_spectrum_1d(self):
        """Test that phase_randomize preserves power spectrum for 1D data.

        This is THE CRITICAL property - power spectrum must be preserved exactly.
        """
        # Use shorter signal for tier1 speed (still sufficient for FFT)
        data = np.random.randn(100)  # Reduced from 200 for tier1 speed
        randomized = phase_randomize(data, random_state=42)

        # Compute power spectra
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2

        # Should match exactly (within numerical precision)
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    def test_preserves_power_spectrum_2d(self):
        """Test that phase_randomize preserves power spectrum for 2D data."""
        data = np.random.randn(100, 5)  # Reduced from 200, 5 for tier1 speed
        randomized = phase_randomize(data, random_state=42)

        # Check each feature independently
        for i in range(data.shape[1]):
            power_orig = np.abs(np.fft.rfft(data[:, i])) ** 2
            power_rand = np.abs(np.fft.rfft(randomized[:, i])) ** 2
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    def test_changes_phase_1d(self):
        """Test that phase_randomize actually changes the signal."""
        # Use deterministic signal
        t = np.linspace(0, 10 * np.pi, 50)  # Reduced from 100 for tier1 speed
        data = np.sin(t)

        randomized = phase_randomize(data, random_state=42)

        # Signal should be different
        assert not np.allclose(data, randomized)

    def test_changes_phase_2d(self):
        """Test that phase_randomize changes signals for 2D data."""
        data = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
        randomized = phase_randomize(data, random_state=42)

        # Should be different
        assert not np.allclose(data, randomized)

    def test_deterministic_with_seed_1d(self):
        """Test that phase_randomize is deterministic with random_state for 1D."""
        data = np.random.randn(50)  # Reduced from 100 for tier1 speed
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)

    def test_deterministic_with_seed_2d(self):
        """Test that phase_randomize is deterministic with random_state for 2D."""
        data = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)

    def test_backend_consistency_numpy(self):
        """Test that phase_randomize works with NumPy backend."""
        data = np.random.randn(50)  # Reduced from 100 for tier1 speed
        randomized = phase_randomize(data, device="cpu", random_state=42)

        # Should preserve power spectrum
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)

    @pytest.mark.slow
    def test_backend_consistency_torch_1d(self):
        """Test that phase_randomize works with torch backend for 1D data."""
        pytest.importorskip("torch")
        from nltools.algorithms.backends import check_gpu_available

        if not check_gpu_available()[0]:
            pytest.skip("GPU not available")

        data = np.random.randn(200)
        randomized = phase_randomize(data, device="gpu", random_state=42)

        # Should preserve power spectrum (within float32 tolerance)
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2
        # GPU uses float32, so relax tolerance compared to NumPy float64
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    def test_backend_consistency_torch_2d(self):
        """Test that phase_randomize works with torch backend for 2D data."""
        pytest.importorskip("torch")
        from nltools.algorithms.backends import check_gpu_available

        if not check_gpu_available()[0]:
            pytest.skip("GPU not available")

        data = np.random.randn(200, 5)
        randomized = phase_randomize(data, device="gpu", random_state=42)

        # Should preserve power spectrum for each feature
        for i in range(data.shape[1]):
            power_orig = np.abs(np.fft.rfft(data[:, i])) ** 2
            power_rand = np.abs(np.fft.rfft(randomized[:, i])) ** 2
            # GPU uses float32, so relax tolerance compared to NumPy float64
            np.testing.assert_allclose(power_orig, power_rand, rtol=1e-4, atol=1e-4)

    @pytest.mark.slow
    def test_backend_consistency_torch_vs_numpy(self):
        """Test that torch and numpy backends produce similar results."""
        pytest.importorskip("torch")
        from nltools.algorithms.backends import check_gpu_available

        if not check_gpu_available()[0]:
            pytest.skip("GPU not available")

        data = np.random.randn(200)
        randomized_numpy = phase_randomize(data, device="cpu", random_state=42)
        randomized_torch = phase_randomize(data, device="gpu", random_state=42)

        # Results should match within float32 tolerance (GPU uses float32)
        np.testing.assert_allclose(
            randomized_numpy, randomized_torch, rtol=1e-5, atol=1e-5
        )


class TestGpuDrawIdentity:
    """GPU permutations must reuse the CPU paths' exact RNG derivations.

    The determinism contract (inference-internals): same seed → same
    permutation draws on every backend; device changes only the arithmetic
    (float32 rounding), never which permutations are evaluated.
    """

    def test_batched_circle_shift_amounts_match_cpu_draw(self):
        from nltools.algorithms.inference.timeseries import _circle_shift_amounts

        rng = np.random.RandomState(0)
        seeds = rng.randint(2**31 - 1, size=8)
        n_samples = 200
        x = np.random.RandomState(1).randn(n_samples)

        amounts = _circle_shift_amounts(seeds, n_samples)
        for seed, amount in zip(seeds, amounts):
            # Same shift circle_shift() itself would draw for this seed —
            # and applying it reproduces circle_shift()'s output exactly.
            expected = circle_shift(x, random_state=seed)
            np.testing.assert_array_equal(np.roll(x, amount), expected)

    def test_batched_phase_randomize_matches_cpu_per_seed(self):
        pytest.importorskip("torch")
        from nltools.algorithms.backends import check_gpu_available
        from nltools.algorithms.inference.timeseries import (
            _phase_randomize_gpu_batched,
        )

        if not check_gpu_available()[0]:
            pytest.skip("no GPU device available")

        backend = Backend("torch")
        n_samples = 200
        x = np.random.RandomState(1).randn(n_samples)
        seeds = np.random.RandomState(0).randint(2**31 - 1, size=6)

        data_device = backend.to_device(x.astype(np.float32))
        batched = backend.to_numpy(
            _phase_randomize_gpu_batched(data_device, seeds, backend, None)
        )
        for i, seed in enumerate(seeds):
            cpu = phase_randomize(x, random_state=seed)
            # float32 device arithmetic vs float64 numpy — same permutation,
            # rounding-level differences only.
            np.testing.assert_allclose(batched[i], cpu, atol=1e-4)


class TestTimeseriesCorrelation:
    """Tests for timeseries_correlation_permutation_test() function."""

    def test_basic_functionality_circle_shift(self):
        """Test basic functionality with circle_shift method."""
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_basic_functionality_phase_randomize(self):
        """Test basic functionality with phase_randomize method."""
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        result = timeseries_correlation_permutation_test(
            x, y, method="phase_randomize", n_permute=100, random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with random_state."""
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

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
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

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

    def test_spearman_metric(self):
        """Test with Spearman correlation metric."""
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
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
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(30)  # Reduced from 50 for tier1 speed
        y = np.random.randn(30)  # Reduced from 50 for tier1 speed

        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=50, metric="kendall", random_state=42
        )

        assert "correlation" in result
        assert "p" in result

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        with pytest.raises(ValueError, match="method must be"):
            timeseries_correlation_permutation_test(
                x, y, method="invalid_method", n_permute=100, random_state=42
            )

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(25)  # Reduced from 50 for tier1 speed

        with pytest.raises(ValueError, match="same length"):
            timeseries_correlation_permutation_test(
                x, y, method="circle_shift", n_permute=100, random_state=42
            )


# ============================================================================
# GPU Timeseries Tests
# ============================================================================


class TestTimeseriesGPU:
    """Tests for GPU-accelerated timeseries permutation tests."""

    pytestmark = pytest.mark.skipif(
        not check_gpu_available()[0], reason="GPU not available"
    )

    @pytest.mark.slow
    def test_gpu_basic_functionality_circle_shift(self):
        """Test basic GPU functionality with circle_shift method."""
        pytest.importorskip("torch")
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, device="gpu", random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert "device" in result
        assert result["device"] == "gpu"
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.slow
    def test_gpu_basic_functionality_phase_randomize(self):
        """Test basic GPU functionality with phase_randomize method."""
        pytest.importorskip("torch")
        from nltools.algorithms import (
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
            device="gpu",
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result
        assert "device" in result
        assert result["device"] == "gpu"
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    @pytest.mark.slow
    def test_gpu_deterministic_with_seed(self):
        """Test that GPU results are deterministic with random_state."""
        pytest.importorskip("torch")
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        result1 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, device="gpu", random_state=42
        )
        result2 = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, device="gpu", random_state=42
        )

        np.testing.assert_equal(result1["correlation"], result2["correlation"])
        np.testing.assert_equal(result1["p"], result2["p"])

    def test_gpu_return_null_distribution(self):
        """Test that GPU returns null distribution when requested."""
        pytest.importorskip("torch")
        import warnings
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Suppress MPS precision warning (expected for torch-mps backend)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="torch-mps backend uses float32", category=UserWarning
            )
            result = timeseries_correlation_permutation_test(
                x,
                y,
                method="circle_shift",
                n_permute=100,
                device="gpu",
                return_null=True,
                random_state=42,
            )

        assert "null_dist" in result
        assert result["null_dist"].shape == (100,)

    @pytest.mark.slow
    def test_gpu_matches_cpu_circle_shift(self):
        """Test that GPU circle_shift matches CPU results (within float32 tolerance)."""
        pytest.importorskip("torch")
        from nltools.algorithms import (
            timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(200)
        y = np.random.randn(200)

        result_cpu = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, device="cpu", random_state=42
        )
        result_gpu = timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, device="gpu", random_state=42
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

    @pytest.mark.slow
    def test_gpu_matches_cpu_phase_randomize(self):
        """Test that GPU phase_randomize matches CPU results (within float32 tolerance)."""
        pytest.importorskip("torch")
        from nltools.algorithms import (
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
            device="cpu",
            random_state=42,
        )
        result_gpu = timeseries_correlation_permutation_test(
            x,
            y,
            method="phase_randomize",
            n_permute=100,
            device="gpu",
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_gpu_circle_shift_correctness(self):
        """Test that GPU circle_shift produces correct results."""
        pytest.importorskip("torch")
        from nltools.algorithms.inference.timeseries import _circle_shift_gpu
        from nltools.algorithms import circle_shift

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

    @pytest.mark.slow
    def test_gpu_batching_prevents_oom(self):
        """Test that GPU batching handles large problems without OOM."""
        torch = pytest.importorskip("torch")

        if not torch.cuda.is_available():
            pytest.skip("GPU not available for OOM test")

        from nltools.algorithms import (
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
            device="gpu",
            max_gpu_memory_gb=1.0,  # Small memory budget to force batching
            random_state=42,
        )

        assert "correlation" in result
        assert "p" in result
        assert result["null_dist"].shape == (5000,)


# ============================================================================
# Test Timeseries Correlation Statistical Correctness
# ============================================================================
