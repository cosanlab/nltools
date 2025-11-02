"""
Tests for voxel-wise ICC computation.

Tests both basic functionality and statistical correctness,
following the pattern used in other inference module tests.
"""

import pytest
import numpy as np

from nltools.algorithms.inference.icc import (
    compute_icc_voxelwise,
    _compute_single_icc,
    _compute_icc_vectorized,
)
from nltools.stats import compute_icc
from nltools.backends import check_gpu_available


class TestComputeICCVoxelwise:
    """Test basic functionality of compute_icc_voxelwise."""

    def test_basic_functionality_single_voxel(self):
        """Test basic ICC computation with single voxel."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 1

        data = np.random.randn(n_subjects * n_sessions, n_voxels)
        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        assert icc_map.shape == (n_voxels,)
        assert isinstance(icc_map[0], (float, np.floating))
        assert -1 <= icc_map[0] <= 1

    def test_basic_functionality_multi_voxel(self):
        """Test basic ICC computation with multiple voxels."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 100

        data = np.random.randn(n_subjects * n_sessions, n_voxels)
        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        assert icc_map.shape == (n_voxels,)
        assert np.all((-1 <= icc_map) & (icc_map <= 1))

    def test_all_icc_types(self):
        """Test that all ICC types work."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 50

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        for icc_type in ["icc1", "icc2", "icc3"]:
            icc_map = compute_icc_voxelwise(
                data, n_subjects, n_sessions, icc_type=icc_type, parallel=None
            )
            assert icc_map.shape == (n_voxels,)
            assert np.all(np.isfinite(icc_map))
            assert np.all((-1 <= icc_map) & (icc_map <= 1))

    def test_correctness_vs_single_icc(self):
        """Test that voxel-wise ICC matches single-voxel compute_icc."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 20

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        # Compute voxel-wise
        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        # Compute single-voxel for each voxel
        for voxel_idx in range(n_voxels):
            Y_single = data[:, voxel_idx].reshape(n_subjects, n_sessions)
            icc_single = compute_icc(Y_single, icc_type="icc2")

            np.testing.assert_almost_equal(icc_map[voxel_idx], icc_single, decimal=10)

    def test_cpu_parallel(self):
        """Test CPU parallelization (for medium-sized problems only)."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 5000  # Falls in parallelization range (1K-10K)

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        # Single-threaded vectorized (default for large problems)
        icc_single = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        # CPU-parallel (should be used for 1K-10K voxels)
        icc_parallel = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel="cpu"
        )

        # Results should match (within numerical precision)
        np.testing.assert_allclose(icc_single, icc_parallel, rtol=1e-5)

    def test_cpu_parallel_auto_n_jobs(self):
        """Test CPU parallelization with auto-detection of n_jobs."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 5000  # Falls in parallelization range (1K-10K)

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        # Test with n_jobs=-1 (auto-detect based on memory)
        icc_auto = compute_icc_voxelwise(
            data,
            n_subjects,
            n_sessions,
            icc_type="icc2",
            parallel="cpu",
            n_jobs=-1,  # Auto-detect
        )

        # Test with explicit n_jobs=2
        icc_explicit = compute_icc_voxelwise(
            data,
            n_subjects,
            n_sessions,
            icc_type="icc2",
            parallel="cpu",
            n_jobs=2,  # Explicit
        )

        # Results should match (within numerical precision)
        np.testing.assert_allclose(icc_auto, icc_explicit, rtol=1e-5)

    def test_cpu_parallel_memory_constraint(self):
        """Test CPU parallelization respects memory constraints."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 5000  # Falls in parallelization range (1K-10K)

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        # Request more workers than memory allows (very restrictive memory limit)
        # With very small memory budget, should reduce workers
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            icc_result = compute_icc_voxelwise(
                data,
                n_subjects,
                n_sessions,
                icc_type="icc2",
                parallel="cpu",
                n_jobs=8,  # Request 8 workers
                max_gpu_memory_gb=0.1,  # Very restrictive (0.1 GB = 100 MB)
            )

            # Should complete successfully (may have reduced workers)
            assert icc_result.shape == (n_voxels,)
            assert np.all(np.isfinite(icc_result))

            # May or may not warn depending on system memory
            # (warning threshold is 20% reduction)
            if len(w) > 0:
                assert any("exceeds memory limit" in str(warning.message) for warning in w)

    @pytest.mark.tier2
    def test_gpu_acceleration(self):
        """Test GPU acceleration (if available)."""
        if not check_gpu_available():
            pytest.skip("GPU not available")

        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 1000  # Enough voxels to benefit from GPU

        data = np.random.randn(n_subjects * n_sessions, n_voxels).astype(np.float32)

        # CPU-parallel
        icc_cpu = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel="cpu"
        )

        # GPU
        icc_gpu = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel="gpu"
        )

        # Results should match (GPU uses float32, so slightly lower precision)
        np.testing.assert_allclose(icc_cpu, icc_gpu, rtol=1e-3, atol=1e-3)

    def test_invalid_data_shape(self):
        """Test that invalid data shape raises error."""
        n_subjects = 10
        n_sessions = 5
        n_voxels = 100

        # Wrong number of images
        data = np.random.randn(n_subjects * n_sessions + 1, n_voxels)
        with pytest.raises(ValueError, match="must equal n_subjects \\* n_sessions"):
            compute_icc_voxelwise(data, n_subjects, n_sessions)

    def test_invalid_icc_type(self):
        """Test that invalid ICC type raises error."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 10

        data = np.random.randn(n_subjects * n_sessions, n_voxels)
        with pytest.raises(ValueError, match="icc_type must be"):
            compute_icc_voxelwise(data, n_subjects, n_sessions, icc_type="invalid")

    def test_invalid_parallel(self):
        """Test that invalid parallel option raises error."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 10

        data = np.random.randn(n_subjects * n_sessions, n_voxels)
        with pytest.raises(ValueError, match="parallel must be"):
            compute_icc_voxelwise(data, n_subjects, n_sessions, parallel="invalid")


class TestICCCorrectness:
    """Test statistical correctness of ICC computation."""

    @pytest.mark.tier1
    def test_icc1_vs_icc3_same_formula(self):
        """Test that ICC1 and ICC3 produce same results (same formula)."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 50

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        icc1 = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc1", parallel=None
        )
        icc3 = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc3", parallel=None
        )

        # ICC1 and ICC3 use the same formula
        np.testing.assert_almost_equal(icc1, icc3, decimal=10)

    @pytest.mark.tier1
    def test_icc2_vs_icc3_relationship(self):
        """Test that ICC2 <= ICC3 when session effects exist."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 50

        # Create data with session effects
        subject_effects = np.linspace(-1, 1, n_subjects)
        session_effects = np.array([2.0, -2.0, 1.0, -1.0, 0.0])
        noise_level = 0.1

        data = np.zeros((n_subjects * n_sessions, n_voxels))
        for i in range(n_subjects):
            for j in range(n_sessions):
                data[i * n_sessions + j, :] = (
                    subject_effects[i]
                    + session_effects[j]
                    + np.random.randn(n_voxels) * noise_level
                )

        icc2 = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )
        icc3 = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc3", parallel=None
        )

        # ICC2 should be <= ICC3 (ICC2 has additional term in denominator)
        assert np.all(icc2 <= icc3 + 1e-10), (
            "ICC2 should be <= ICC3 when session effects exist"
        )

    @pytest.mark.tier1
    def test_perfect_reliability(self):
        """Test ICC with perfect reliability (should be close to 1.0)."""
        np.random.seed(42)
        n_subjects = 5
        n_sessions = 3
        n_voxels = 20

        # Create perfectly reliable data: each subject has identical values across sessions
        subject_effects = np.linspace(-2, 2, n_subjects)
        data = np.zeros((n_subjects * n_sessions, n_voxels))
        for i in range(n_subjects):
            for j in range(n_sessions):
                data[i * n_sessions + j, :] = (
                    subject_effects[i]
                    + np.random.randn(n_voxels) * 0.001  # Minimal noise
                )

        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        # With near-perfect reliability, ICC should be very high (>0.95)
        assert np.all(icc_map > 0.95), (
            f"Perfect reliability should produce ICC > 0.95, got min={icc_map.min():.6f}"
        )

    @pytest.mark.tier1
    def test_zero_reliability(self):
        """Test ICC with zero reliability (pure noise, should be near 0)."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 50

        # Create pure noise data (no systematic subject effects)
        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        # With zero reliability (pure noise), ICC should be near 0 or slightly negative
        assert np.all(icc_map < 0.5), (
            f"Zero reliability should produce ICC < 0.5, got max={icc_map.max():.6f}"
        )

    @pytest.mark.tier1
    def test_effect_size_sensitivity(self):
        """Test that higher reliability produces higher ICC values."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 30

        # Create data with varying reliability levels
        reliability_levels = [
            (1.0, 0.5),  # Low reliability: high noise, low signal
            (0.5, 1.0),  # High reliability: low noise, high signal
        ]

        icc_values = []

        for noise_level, signal_level in reliability_levels:
            subject_effects = np.linspace(-signal_level, signal_level, n_subjects)
            data = np.zeros((n_subjects * n_sessions, n_voxels))
            for i in range(n_subjects):
                for j in range(n_sessions):
                    data[i * n_sessions + j, :] = (
                        subject_effects[i] + np.random.randn(n_voxels) * noise_level
                    )

            icc_map = compute_icc_voxelwise(
                data, n_subjects, n_sessions, icc_type="icc2", parallel=None
            )
            icc_values.append(np.mean(icc_map))

        # Higher reliability should produce higher ICC
        assert icc_values[1] > icc_values[0], (
            f"Higher reliability should produce higher ICC. "
            f"Low: {icc_values[0]:.6f}, High: {icc_values[1]:.6f}"
        )


class TestICCEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_data(self):
        """Test ICC with constant data (all values the same)."""
        n_subjects = 5
        n_sessions = 3
        n_voxels = 10

        # All values identical
        data = np.ones((n_subjects * n_sessions, n_voxels))

        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        # With constant data, ICC may be NaN or finite (depending on implementation)
        assert np.all(np.isfinite(icc_map) | np.isnan(icc_map)), (
            "ICC should be finite or NaN for constant data"
        )

    def test_small_sample_size(self):
        """Test ICC with minimal sample size."""
        np.random.seed(42)
        n_subjects = 3
        n_sessions = 2
        n_voxels = 5

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        assert icc_map.shape == (n_voxels,)
        assert np.all(np.isfinite(icc_map))

    def test_large_voxel_count(self):
        """Test ICC with large voxel count (uses vectorized, memory efficient)."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 50000  # Large voxel count (uses vectorized, not parallel)

        data = np.random.randn(n_subjects * n_sessions, n_voxels)

        # For large voxel counts, vectorized (parallel=None) is memory efficient
        icc_map = compute_icc_voxelwise(
            data, n_subjects, n_sessions, icc_type="icc2", parallel=None
        )

        assert icc_map.shape == (n_voxels,)
        assert np.all(np.isfinite(icc_map))
        assert np.all((-1 <= icc_map) & (icc_map <= 1))


class TestICSingleVoxel:
    """Test _compute_single_icc helper function."""

    def test_single_icc_matches_compute_icc(self):
        """Test that _compute_single_icc matches compute_icc."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5

        Y = np.random.randn(n_subjects, n_sessions)

        for icc_type in ["icc1", "icc2", "icc3"]:
            icc_single = _compute_single_icc(Y, icc_type)
            icc_stats = compute_icc(Y, icc_type=icc_type)

            np.testing.assert_almost_equal(icc_single, icc_stats, decimal=10)

    def test_single_icc_edge_cases(self):
        """Test _compute_single_icc with edge cases."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5

        # Constant data
        Y_constant = np.ones((n_subjects, n_sessions))
        icc_constant = _compute_single_icc(Y_constant, "icc2")
        assert np.isfinite(icc_constant) or np.isnan(icc_constant)

        # Perfect reliability
        subject_effects = np.linspace(-2, 2, n_subjects)
        Y_perfect = np.zeros((n_subjects, n_sessions))
        for i in range(n_subjects):
            Y_perfect[i, :] = subject_effects[i]
        icc_perfect = _compute_single_icc(Y_perfect, "icc2")
        assert icc_perfect > 0.99


class TestICCVectorized:
    """Test vectorized ICC computation."""

    def test_vectorized_matches_single_voxel(self):
        """Test that vectorized computation matches single-voxel computation."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 20

        Y = np.random.randn(n_subjects, n_sessions, n_voxels)

        # Vectorized
        icc_vectorized = _compute_icc_vectorized(Y, "icc2")

        # Single-voxel for each
        icc_single = []
        for voxel_idx in range(n_voxels):
            Y_voxel = Y[:, :, voxel_idx]
            icc_single.append(_compute_single_icc(Y_voxel, "icc2"))

        np.testing.assert_almost_equal(icc_vectorized, np.array(icc_single), decimal=10)

    def test_vectorized_all_icc_types(self):
        """Test vectorized computation for all ICC types."""
        np.random.seed(42)
        n_subjects = 10
        n_sessions = 5
        n_voxels = 30

        Y = np.random.randn(n_subjects, n_sessions, n_voxels)

        for icc_type in ["icc1", "icc2", "icc3"]:
            icc_map = _compute_icc_vectorized(Y, icc_type)
            assert icc_map.shape == (n_voxels,)
            assert np.all(np.isfinite(icc_map))
            assert np.all((-1 <= icc_map) & (icc_map <= 1))
