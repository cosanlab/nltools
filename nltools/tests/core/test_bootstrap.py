"""Tests for bootstrap inference utilities."""

import numpy as np
from nltools.algorithms.inference.bootstrap import (
    OnlineBootstrapStats,
    _bootstrap_simple_cpu_parallel,
    _bootstrap_ridge_weights_cpu_parallel,
)


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


class TestBootstrapSimpleMethods:
    """Test suite for simple bootstrap methods."""

    def test_bootstrap_simple_methods_all(self):
        """Test all simple aggregation methods work correctly."""
        np.random.seed(42)
        data = np.random.randn(50, 20)  # 50 samples, 20 features

        methods = ["mean", "median", "std", "sum", "min", "max"]

        for method in methods:
            result = _bootstrap_simple_cpu_parallel(
                data, method, n_samples=100, n_jobs=1, random_state=42
            )

            # Check dict structure
            assert "mean" in result
            assert "std" in result
            assert "Z" in result
            assert "p" in result
            assert "ci_lower" in result
            assert "ci_upper" in result
            assert "backend" in result
            assert "samples" not in result  # save_weights=False

            # Check shapes
            assert result["mean"].shape == (20,)
            assert result["std"].shape == (20,)
            assert result["Z"].shape == (20,)
            assert result["p"].shape == (20,)

            # Check values are reasonable
            assert not np.any(np.isnan(result["mean"]))
            assert not np.any(np.isnan(result["std"]))
            assert np.all(result["std"] >= 0)

    def test_bootstrap_simple_reproducibility(self):
        """Test same random_state produces identical results."""
        np.random.seed(42)
        data = np.random.randn(50, 20)

        result1 = _bootstrap_simple_cpu_parallel(
            data, "mean", n_samples=100, n_jobs=1, random_state=42
        )
        result2 = _bootstrap_simple_cpu_parallel(
            data, "mean", n_samples=100, n_jobs=1, random_state=42
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(result1["mean"], result2["mean"])
        np.testing.assert_array_almost_equal(result1["std"], result2["std"])
        np.testing.assert_array_almost_equal(result1["Z"], result2["Z"])
        np.testing.assert_array_almost_equal(result1["p"], result2["p"])

    def test_bootstrap_simple_save_weights(self):
        """Test save_weights=True stores all samples."""
        np.random.seed(42)
        data = np.random.randn(50, 20)

        n_samples = 100
        result = _bootstrap_simple_cpu_parallel(
            data,
            "mean",
            n_samples=n_samples,
            save_weights=True,
            n_jobs=1,
            random_state=42,
        )

        # Samples should be stored
        assert "samples" in result
        assert result["samples"].shape == (n_samples, 20)

        # Verify percentile CIs match samples
        ci_lower_from_samples = np.percentile(result["samples"], 2.5, axis=0)
        ci_upper_from_samples = np.percentile(result["samples"], 97.5, axis=0)

        np.testing.assert_array_almost_equal(
            result["ci_lower"], ci_lower_from_samples, decimal=5
        )
        np.testing.assert_array_almost_equal(
            result["ci_upper"], ci_upper_from_samples, decimal=5
        )

    def test_bootstrap_simple_progress(self):
        """Test progress bar works (basic smoke test)."""
        np.random.seed(42)
        data = np.random.randn(50, 20)

        # Just run and verify no errors
        result = _bootstrap_simple_cpu_parallel(
            data, "mean", n_samples=50, n_jobs=1, random_state=42
        )

        assert result is not None
        assert "mean" in result


class TestBootstrapRidgeWeights:
    """Test suite for Ridge model weights bootstrap."""

    def test_bootstrap_ridge_weights_efficient(self):
        """Test Ridge weights bootstrap in efficient mode."""
        np.random.seed(42)
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        y = np.random.randn(100, 50)  # 50 voxels
        alpha = 1.0

        result = _bootstrap_ridge_weights_cpu_parallel(
            X, y, alpha, n_samples=100, n_jobs=1, random_state=42
        )

        # Check dict structure
        assert "mean" in result
        assert "std" in result
        assert "Z" in result
        assert "p" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "backend" in result
        assert "samples" not in result  # save_weights=False

        # Check shapes (weights are n_features × n_voxels)
        assert result["mean"].shape == (10, 50)
        assert result["std"].shape == (10, 50)

        # Check no NaNs
        assert not np.any(np.isnan(result["mean"]))
        assert not np.any(np.isnan(result["std"]))
        assert np.all(result["std"] >= 0)

    def test_bootstrap_ridge_weights_full(self):
        """Test Ridge weights bootstrap with save_weights=True."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 50)
        alpha = 1.0
        n_samples = 100

        result = _bootstrap_ridge_weights_cpu_parallel(
            X,
            y,
            alpha,
            n_samples=n_samples,
            save_weights=True,
            n_jobs=1,
            random_state=42,
        )

        # Samples should be stored
        assert "samples" in result
        assert result["samples"].shape == (n_samples, 10, 50)

        # Verify percentile CIs match samples
        ci_lower_from_samples = np.percentile(result["samples"], 2.5, axis=0)
        ci_upper_from_samples = np.percentile(result["samples"], 97.5, axis=0)

        np.testing.assert_array_almost_equal(
            result["ci_lower"], ci_lower_from_samples, decimal=5
        )
        np.testing.assert_array_almost_equal(
            result["ci_upper"], ci_upper_from_samples, decimal=5
        )

    def test_bootstrap_ridge_weights_variance(self):
        """Test that bootstrap samples have variance."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 50)
        alpha = 1.0

        result = _bootstrap_ridge_weights_cpu_parallel(
            X, y, alpha, n_samples=100, n_jobs=1, random_state=42
        )

        # Bootstrap std should be > 0 for most weights
        # (Some might be near-zero if feature is uninformative)
        assert np.mean(result["std"] > 0) > 0.95  # At least 95% have variance
        assert np.mean(result["std"] > 1e-6) > 0.90  # Most have meaningful variance

    def test_bootstrap_ridge_weights_preserves_alpha(self):
        """Test that alpha parameter affects results."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 50)

        # Try two different alpha values
        result1 = _bootstrap_ridge_weights_cpu_parallel(
            X, y, alpha=0.1, n_samples=50, n_jobs=1, random_state=42
        )
        result2 = _bootstrap_ridge_weights_cpu_parallel(
            X, y, alpha=10.0, n_samples=50, n_jobs=1, random_state=42
        )

        # Results should be different (different regularization)
        # Higher alpha = more shrinkage = smaller weights
        assert not np.allclose(result1["mean"], result2["mean"])
        # Check that higher alpha generally produces smaller weights
        assert np.mean(np.abs(result2["mean"])) < np.mean(np.abs(result1["mean"]))

    def test_bootstrap_ridge_weights_memory(self):
        """Test memory efficiency."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 50)
        alpha = 1.0

        # Efficient mode
        result_efficient = _bootstrap_ridge_weights_cpu_parallel(
            X, y, alpha, n_samples=100, save_weights=False, n_jobs=1, random_state=42
        )

        # Full mode
        result_full = _bootstrap_ridge_weights_cpu_parallel(
            X, y, alpha, n_samples=100, save_weights=True, n_jobs=1, random_state=42
        )

        # Calculate approximate memory usage
        import sys

        # Efficient mode: Only stores mean, std, etc. (6 arrays of shape (10, 50))
        efficient_size = sum(
            sys.getsizeof(v)
            for v in result_efficient.values()
            if isinstance(v, np.ndarray)
        )

        # Full mode: Also stores samples (100, 10, 50)
        full_size = sum(
            sys.getsizeof(v) for v in result_full.values() if isinstance(v, np.ndarray)
        )

        # Full mode should be much larger (stores all samples)
        # Samples alone are 100×(10×50) = 50,000 floats = 400KB
        # Other results are ~6×500 = 3,000 floats = 24KB
        # Ratio should be ~17× or more
        assert full_size > efficient_size * 10
