"""Tests for timeseries permutation tests and helper functions."""

import pytest
import numpy as np

from nltools.algorithms import circle_shift, phase_randomize


class TestCircleShift:
    """Tests for circle_shift() function."""

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


class TestPhaseRandomize:
    """Tests for phase_randomize() function."""

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

    def test_deterministic_with_seed_2d(self):
        """Test that phase_randomize is deterministic with random_state for 2D."""
        data = np.random.randn(50, 5)  # Reduced from 100, 5 for tier1 speed
        rand1 = phase_randomize(data, random_state=42)
        rand2 = phase_randomize(data, random_state=42)
        np.testing.assert_array_equal(rand1, rand2)


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
