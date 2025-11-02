"""
Tests for BrainData.icc() integration with voxel-wise ICC computation.
"""

import pytest
import numpy as np

from nltools.data import BrainData
from nltools.stats import compute_icc


class TestBrainDataICC:
    """Test BrainData.icc() method integration."""

    def test_basic_functionality(self, minimal_brain_data):
        """Test basic ICC computation with BrainData."""
        # Create test-retest data: 5 subjects × 3 sessions
        n_subjects = 5
        n_sessions = 3

        # Create synthetic data with subject effects
        np.random.seed(42)
        subject_effects = np.linspace(-1, 1, n_subjects)

        # Create data array directly
        n_voxels = minimal_brain_data.shape[1]
        data_array = np.zeros((n_subjects * n_sessions, n_voxels))
        for i in range(n_subjects):
            for j in range(n_sessions):
                data_array[i * n_sessions + j, :] = (
                    subject_effects[i] + np.random.randn(n_voxels) * 0.1
                )

        # Create BrainData from array
        test_data = minimal_brain_data.copy()
        test_data.data = data_array

        # Compute ICC
        icc_map = test_data.icc(n_subjects=n_subjects, n_sessions=n_sessions)

        # Check output
        assert isinstance(icc_map, BrainData)
        assert icc_map.shape == (1, minimal_brain_data.shape[1])
        assert np.all(np.isfinite(icc_map.data))
        assert np.all((-1 <= icc_map.data) & (icc_map.data <= 1))

    def test_correctness_vs_single_voxel(self):
        """Test that BrainData.icc() matches single-voxel compute_icc."""
        from nltools.simulator import Simulator

        # Create test data: 10 subjects × 5 sessions
        n_subjects = 10
        n_sessions = 5
        sim = Simulator()
        dat = sim.create_data([0, 1], 1, reps=n_subjects * n_sessions)
        # Reshape to have correct number of images
        dat.data = dat.data[: n_subjects * n_sessions]

        # Compute ICC with BrainData
        icc_map = dat.icc(n_subjects=n_subjects, n_sessions=n_sessions, parallel=None)

        # For a few voxels, verify against single-voxel computation
        n_test_voxels = min(10, dat.shape[1])
        for voxel_idx in range(n_test_voxels):
            Y_single = dat.data[:, voxel_idx].reshape(n_subjects, n_sessions)
            icc_single = compute_icc(Y_single, icc_type="icc2")

            # Allow for float32 precision differences
            np.testing.assert_allclose(
                icc_map.data[0, voxel_idx], icc_single, rtol=1e-5, atol=1e-6
            )

    def test_all_icc_types(self):
        """Test that all ICC types work with BrainData."""
        from nltools.simulator import Simulator

        n_subjects = 5
        n_sessions = 3
        sim = Simulator()
        dat = sim.create_data([0, 1], 1, reps=n_subjects * n_sessions)
        dat.data = dat.data[: n_subjects * n_sessions]

        for icc_type in ["icc1", "icc2", "icc3"]:
            icc_map = dat.icc(
                n_subjects=n_subjects,
                n_sessions=n_sessions,
                icc_type=icc_type,
                parallel=None,
            )

            assert isinstance(icc_map, BrainData)
            assert icc_map.shape == (1, dat.shape[1])
            assert np.all(np.isfinite(icc_map.data))
            assert np.all((-1 <= icc_map.data) & (icc_map.data <= 1))

    def test_cpu_parallel(self):
        """Test CPU parallelization with BrainData (for medium-sized problems)."""
        from nltools.simulator import Simulator

        n_subjects = 5
        n_sessions = 3
        sim = Simulator()
        dat = sim.create_data([0, 1], 1, reps=n_subjects * n_sessions)
        dat.data = dat.data[: n_subjects * n_sessions]

        # Use a subset of voxels (5K) to test parallelization (falls in 1K-10K range)
        # This tests the parallelization path without using full brain data
        n_test_voxels = 5000
        dat_subset = dat.copy()
        dat_subset.data = dat.data[:, :n_test_voxels]

        # Single-threaded vectorized (default)
        icc_single = dat_subset.icc(
            n_subjects=n_subjects,
            n_sessions=n_sessions,
            parallel=None,
        )

        # CPU-parallel (should be used for 1K-10K voxels)
        icc_parallel = dat_subset.icc(
            n_subjects=n_subjects,
            n_sessions=n_sessions,
            parallel="cpu",
        )

        # Results should match (within numerical precision)
        # Allow for float32 precision differences in parallel computation
        np.testing.assert_allclose(
            icc_single.data, icc_parallel.data, rtol=1e-4, atol=1e-6
        )

    def test_invalid_shape(self):
        """Test that invalid shape raises error."""
        from nltools.simulator import Simulator

        n_subjects = 5
        n_sessions = 3
        sim = Simulator()
        dat = sim.create_data([0, 1], 1, reps=n_subjects * n_sessions)
        dat.data = dat.data[: n_subjects * n_sessions]

        # Wrong number of subjects
        with pytest.raises(ValueError, match="must equal n_subjects \\* n_sessions"):
            dat.icc(n_subjects=10, n_sessions=3)

        # Wrong number of sessions
        with pytest.raises(ValueError, match="must equal n_subjects \\* n_sessions"):
            dat.icc(n_subjects=5, n_sessions=10)

    def test_return_type(self):
        """Test that ICC returns BrainData object."""
        from nltools.simulator import Simulator

        n_subjects = 5
        n_sessions = 3
        sim = Simulator()
        dat = sim.create_data([0, 1], 1, reps=n_subjects * n_sessions)
        dat.data = dat.data[: n_subjects * n_sessions]

        icc_map = dat.icc(n_subjects=n_subjects, n_sessions=n_sessions)

        # Should return BrainData with shape (1, n_voxels)
        assert isinstance(icc_map, BrainData)
        assert icc_map.shape == (1, dat.shape[1])
        assert icc_map.mask is not None  # Should preserve mask

    def test_perfect_reliability(self):
        """Test ICC with perfect reliability data."""
        from nltools.simulator import Simulator

        n_subjects = 5
        n_sessions = 3
        sim = Simulator()
        dat = sim.create_data([0, 1], 1, reps=n_subjects * n_sessions)
        dat.data = dat.data[: n_subjects * n_sessions]

        # Create perfectly reliable data: each subject has identical values across sessions
        subject_effects = np.linspace(-2, 2, n_subjects)
        for i in range(n_subjects):
            for j in range(n_sessions):
                dat.data[i * n_sessions + j, :] = subject_effects[i]

        icc_map = dat.icc(n_subjects=n_subjects, n_sessions=n_sessions, parallel=None)

        # With perfect reliability, ICC should be very high (>0.99)
        assert np.all(icc_map.data > 0.99), (
            f"Perfect reliability should produce ICC > 0.99, got min={icc_map.data.min():.6f}"
        )
