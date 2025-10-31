"""Tests for bootstrap inference utilities."""

import numpy as np
from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats


class TestOnlineBootstrapStats:
    """Test suite for OnlineBootstrapStats class."""

    def test_online_stats_1d_array(self):
        """Test basic aggregation with 1D arrays."""
        # Create stats aggregator for 1D arrays
        stats = OnlineBootstrapStats(shape=(10,), save_samples=False)

        # Update with 100 known samples (from N(0, 1))
        np.random.seed(42)
        for _ in range(100):
            sample = np.random.randn(10)
            stats.update(sample)

        # Get results
        result = stats.get_results()

        # Verify output structure
        assert "mean" in result
        assert "std" in result
        assert "Z" in result
        assert "p" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "samples" not in result  # Efficient mode

        # Verify shapes
        assert result["mean"].shape == (10,)
        assert result["std"].shape == (10,)
        assert result["Z"].shape == (10,)
        assert result["p"].shape == (10,)

        # Verify mean is close to 0 (samples from N(0,1))
        assert np.allclose(result["mean"], 0, atol=0.5)

        # Verify CIs are reasonable
        assert np.all(result["ci_upper"] > result["ci_lower"])

    def test_online_stats_2d_array(self):
        """Test aggregation with 2D arrays (multi-feature)."""
        # Create stats for weight matrix shape (n_features, n_voxels)
        stats = OnlineBootstrapStats(shape=(5, 20), save_samples=False)

        # Update with 100 random samples
        np.random.seed(42)
        for _ in range(100):
            sample = np.random.randn(5, 20)
            stats.update(sample)

        # Get results
        result = stats.get_results()

        # Verify all outputs have correct shape
        assert result["mean"].shape == (5, 20)
        assert result["std"].shape == (5, 20)
        assert result["Z"].shape == (5, 20)
        assert result["p"].shape == (5, 20)
        assert result["ci_lower"].shape == (5, 20)
        assert result["ci_upper"].shape == (5, 20)

        # Verify no samples stored (efficient mode)
        assert "samples" not in result

    def test_online_stats_confidence_intervals(self):
        """Test CI computation using normal approximation."""
        # Create stats aggregator
        stats = OnlineBootstrapStats(
            shape=(10,), save_samples=False, percentiles=(2.5, 97.5)
        )

        # Update with samples from known distribution N(5, 2)
        np.random.seed(42)
        for _ in range(200):
            sample = np.random.randn(10) * 2 + 5  # mean=5, std=2
            stats.update(sample)

        result = stats.get_results()

        # Verify CI relationships
        assert np.all(result["ci_upper"] > result["mean"])
        assert np.all(result["mean"] > result["ci_lower"])

        # Verify CI width is approximately 2 * 1.96 * std (for 95% CI)
        # Width should be close to 3.92 * std
        ci_width = result["ci_upper"] - result["ci_lower"]
        expected_width = 2 * 1.96 * result["std"]
        assert np.allclose(ci_width, expected_width, rtol=0.01)

    def test_online_stats_sample_storage(self):
        """Test sample storage mode for exact percentile CIs."""
        # Create stats with sample storage
        stats = OnlineBootstrapStats(
            shape=(5,), save_samples=True, percentiles=(2.5, 97.5)
        )

        # Track samples manually for verification
        samples_list = []
        np.random.seed(42)
        for _ in range(100):
            sample = np.random.randn(5)
            stats.update(sample)
            samples_list.append(sample)

        result = stats.get_results()

        # Verify samples are stored
        assert "samples" in result
        assert result["samples"].shape == (100, 5)

        # Verify exact percentile CIs match numpy.percentile
        samples_array = np.array(samples_list)
        expected_lower = np.percentile(samples_array, 2.5, axis=0)
        expected_upper = np.percentile(samples_array, 97.5, axis=0)

        assert np.allclose(result["ci_lower"], expected_lower, rtol=1e-10)
        assert np.allclose(result["ci_upper"], expected_upper, rtol=1e-10)

    def test_online_stats_numerical_stability(self):
        """Test Welford's algorithm handles large values correctly."""
        # Create stats aggregator
        stats = OnlineBootstrapStats(shape=(3,), save_samples=False)

        # Create samples with large mean (1e10) and small variance
        # This tests for catastrophic cancellation
        np.random.seed(42)
        mean_val = 1e10
        samples_list = []
        for _ in range(1000):
            sample = np.array([mean_val, mean_val, mean_val]) + np.random.randn(3) * 0.1
            stats.update(sample)
            samples_list.append(sample)

        result = stats.get_results()

        # Verify mean is accurate despite large value
        assert np.allclose(result["mean"], mean_val, rtol=1e-8)

        # Verify small variance is preserved (not lost to numerical errors)
        assert np.all(result["std"] < 1.0)  # Should be ~0.1

        # Compare to numpy.std as ground truth
        samples_array = np.array(samples_list)
        expected_std = np.std(samples_array, axis=0, ddof=1)  # Sample std
        assert np.allclose(result["std"], expected_std, rtol=1e-5)
