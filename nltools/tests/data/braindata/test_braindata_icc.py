"""
Tests for BrainData.icc() facade — verifies the method delegates correctly
to compute_icc_voxelwise(). Numerical correctness of the ICC computation
itself is tested in nltools/tests/core/test_inference/test_icc.py.
"""

import pytest
import numpy as np

from nltools.data import BrainData


def _make_icc_brain_data(
    minimal_brain_data, n_subjects=5, n_sessions=3, reliability="moderate"
):
    """Build ICC test data from minimal_brain_data fixture.

    Args:
        reliability: "perfect" for ICC~1, "moderate" for realistic ICC values.
    """
    n_voxels = minimal_brain_data.shape[1]
    n_images = n_subjects * n_sessions
    np.random.seed(42)

    subject_effects = np.linspace(-2, 2, n_subjects)

    data = np.zeros((n_images, n_voxels))
    for i in range(n_subjects):
        for j in range(n_sessions):
            noise = 0.0 if reliability == "perfect" else np.random.randn(n_voxels) * 0.3
            data[i * n_sessions + j, :] = subject_effects[i] + noise

    bd = minimal_brain_data.copy()
    bd.data = data
    return bd


class TestBrainDataICC:
    """Test BrainData.icc() method integration."""

    def test_basic_functionality(self, minimal_brain_data):
        """Test ICC returns correct shape and value range."""
        bd = _make_icc_brain_data(minimal_brain_data)
        icc_map = bd.icc(n_subjects=5, n_sessions=3)

        assert isinstance(icc_map, BrainData)
        assert icc_map.shape == (1, minimal_brain_data.shape[1])
        assert icc_map.mask is not None
        assert np.all(np.isfinite(icc_map.data))
        assert np.all((-1 <= icc_map.data) & (icc_map.data <= 1))

    @pytest.mark.parametrize("icc_type", ["icc1", "icc2", "icc3"])
    def test_icc_types(self, minimal_brain_data, icc_type):
        """Test that all ICC types work and return valid results."""
        bd = _make_icc_brain_data(minimal_brain_data)
        icc_map = bd.icc(n_subjects=5, n_sessions=3, icc_type=icc_type, parallel=None)

        assert isinstance(icc_map, BrainData)
        assert icc_map.shape == (1, minimal_brain_data.shape[1])
        assert np.all(np.isfinite(icc_map.data))

    def test_perfect_reliability(self, minimal_brain_data):
        """Test ICC ≈ 1 when subjects are perfectly consistent across sessions."""
        bd = _make_icc_brain_data(minimal_brain_data, reliability="perfect")
        icc_map = bd.icc(n_subjects=5, n_sessions=3, parallel=None)

        assert np.all(icc_map.data > 0.99), (
            f"Perfect reliability should give ICC > 0.99, got min={icc_map.data.min():.4f}"
        )

    def test_invalid_shape(self, minimal_brain_data):
        """Test error when n_subjects * n_sessions != n_images."""
        bd = _make_icc_brain_data(minimal_brain_data)

        with pytest.raises(ValueError, match="must equal n_subjects \\* n_sessions"):
            bd.icc(n_subjects=10, n_sessions=3)

        with pytest.raises(ValueError, match="must equal n_subjects \\* n_sessions"):
            bd.icc(n_subjects=5, n_sessions=10)

    def test_cpu_parallel_matches_serial(self, minimal_brain_data):
        """Test CPU parallelization produces same results as serial."""
        bd = _make_icc_brain_data(minimal_brain_data)

        icc_serial = bd.icc(n_subjects=5, n_sessions=3, parallel=None)
        icc_parallel = bd.icc(n_subjects=5, n_sessions=3, parallel="cpu")

        np.testing.assert_allclose(
            icc_serial.data, icc_parallel.data, rtol=1e-4, atol=1e-6
        )
