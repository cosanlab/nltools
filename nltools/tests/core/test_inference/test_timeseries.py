"""Tests for timeseries permutation tests and helper functions."""

import pytest
import numpy as np

from nltools.algorithms import circle_shift, phase_randomize


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
        randomized = phase_randomize(data, random_state=42)

        # Should preserve power spectrum
        power_orig = np.abs(np.fft.rfft(data)) ** 2
        power_rand = np.abs(np.fft.rfft(randomized)) ** 2
        np.testing.assert_allclose(power_orig, power_rand, rtol=1e-10)


class TestTimeseriesCorrelation:
    """Tests for _timeseries_correlation_permutation_test() function."""

    def test_basic_functionality_circle_shift(self):
        """Test basic functionality with circle_shift method."""
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        result = _timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_basic_functionality_phase_randomize(self):
        """Test basic functionality with phase_randomize method."""
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        result = _timeseries_correlation_permutation_test(
            x, y, method="phase_randomize", n_permute=100, random_state=42
        )

        assert "correlation" in result
        assert "p" in result
        assert isinstance(result["correlation"], (float, np.floating))
        assert 0 <= result["p"] <= 1

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with random_state."""
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        result1 = _timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )
        result2 = _timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=100, random_state=42
        )

        np.testing.assert_equal(result1["correlation"], result2["correlation"])
        np.testing.assert_equal(result1["p"], result2["p"])

    def test_return_null_distribution(self):
        """Test that null distribution is returned when requested."""
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        result = _timeseries_correlation_permutation_test(
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
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = x**2  # Nonlinear monotonic relationship

        result = _timeseries_correlation_permutation_test(
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
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(30)  # Reduced from 50 for tier1 speed
        y = np.random.randn(30)  # Reduced from 50 for tier1 speed

        result = _timeseries_correlation_permutation_test(
            x, y, method="circle_shift", n_permute=50, metric="kendall", random_state=42
        )

        assert "correlation" in result
        assert "p" in result

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(50)  # Reduced from 100 for tier1 speed

        with pytest.raises(ValueError, match="method must be"):
            _timeseries_correlation_permutation_test(
                x, y, method="invalid_method", n_permute=100, random_state=42
            )

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        from nltools.algorithms.inference import (
            _timeseries_correlation_permutation_test,
        )

        np.random.seed(42)
        x = np.random.randn(50)  # Reduced from 100 for tier1 speed
        y = np.random.randn(25)  # Reduced from 50 for tier1 speed

        with pytest.raises(ValueError, match="same length"):
            _timeseries_correlation_permutation_test(
                x, y, method="circle_shift", n_permute=100, random_state=42
            )


# ============================================================================
# GPU Timeseries Tests
# ============================================================================


# ============================================================================
# Test Timeseries Correlation Statistical Correctness
# ============================================================================
