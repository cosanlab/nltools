import numpy as np
import pytest

from nltools.data import BrainData, Adjacency


class TestBrainDataCore:
    """Test BrainData dunders, properties, arithmetic, stats, indexing, append, copy, and distance."""

    # ==================== Properties & Statistics ====================

    def test_shape(self, minimal_brain_data):
        """Test shape property returns correct dimensions."""
        assert minimal_brain_data.shape == (50, 5)

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_stat_aggregation(self, minimal_brain_data, method):
        """Test mean/median across axes."""
        func = getattr(minimal_brain_data, method)
        # Default axis=0: aggregate over images
        assert func().shape[0] == minimal_brain_data.shape[1]
        # axis=1: aggregate over voxels
        assert len(func(axis=1)) == minimal_brain_data.shape[0]
        # Invalid axis type
        with pytest.raises(ValueError):
            func(axis="1")
        # Single image returns scalar
        assert isinstance(
            getattr(minimal_brain_data[0], method)(), (float, np.floating)
        )

    def test_std(self, minimal_brain_data):
        """Test standard deviation computation."""
        assert minimal_brain_data.std().shape[0] == minimal_brain_data.shape[1]

    def test_sum(self, minimal_brain_data):
        """Test sum aggregation."""
        s = minimal_brain_data.sum()
        assert s.shape == minimal_brain_data[0].shape

    # ==================== Arithmetic Operations ====================

    def test_add(self, minimal_brain_data):
        """Test addition of BrainData objects and scalars."""
        new = minimal_brain_data + minimal_brain_data
        assert new.shape == minimal_brain_data.shape
        value = 10
        assert (value + minimal_brain_data[0]).mean() == (
            minimal_brain_data[0] + value
        ).mean()

    def test_subtract(self, minimal_brain_data):
        """Test subtraction of BrainData objects and scalars."""
        new = minimal_brain_data - minimal_brain_data
        assert new.shape == minimal_brain_data.shape
        value = 10
        assert (-value - (-1) * minimal_brain_data[0]).mean() == (
            minimal_brain_data[0] - value
        ).mean()

    def test_multiply(self, minimal_brain_data):
        """Test multiplication of BrainData objects, scalars, and arrays."""
        new = minimal_brain_data * minimal_brain_data
        assert new.shape == minimal_brain_data.shape
        value = 10
        assert (value * minimal_brain_data[0]).mean() == (
            minimal_brain_data[0] * value
        ).mean()
        c1 = [0.5, 0.5, -0.5, -0.5]
        new = minimal_brain_data[0:4] * c1
        new2 = (
            minimal_brain_data[0] * 0.5
            + minimal_brain_data[1] * 0.5
            - minimal_brain_data[2] * 0.5
            - minimal_brain_data[3] * 0.5
        )
        np.testing.assert_almost_equal((new - new2).sum(), 0, decimal=4)

    def test_divide(self, minimal_brain_data):
        """Test division of BrainData objects and scalars."""
        new = minimal_brain_data / minimal_brain_data
        assert new.shape == minimal_brain_data.shape
        np.testing.assert_almost_equal(new.mean(axis=0).mean(), 1, decimal=6)
        value = 10
        new2 = minimal_brain_data / value
        np.testing.assert_almost_equal(
            ((new2 * value) - new2).mean().mean(), 0, decimal=2
        )

    def test_inplace_add(self, minimal_brain_data):
        """Test in-place addition with scalars and BrainData."""
        bd = minimal_brain_data[0].copy()
        original_data = bd.data.copy()
        bd += 5
        assert np.allclose(bd.data, original_data + 5)

        bd1 = minimal_brain_data[0].copy()
        bd2 = minimal_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 += bd2
        assert np.allclose(bd1.data, original_data + bd2.data)

    def test_inplace_subtract(self, minimal_brain_data):
        """Test in-place subtraction with scalars and BrainData."""
        bd = minimal_brain_data[0].copy()
        original_data = bd.data.copy()
        bd -= 3
        assert np.allclose(bd.data, original_data - 3)

        bd1 = minimal_brain_data[0].copy()
        bd2 = minimal_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 -= bd2
        assert np.allclose(bd1.data, original_data - bd2.data)

    def test_inplace_multiply(self, minimal_brain_data):
        """Test in-place multiplication with scalars, BrainData, and arrays."""
        bd = minimal_brain_data[0].copy()
        original_data = bd.data.copy()
        bd *= 2
        assert np.allclose(bd.data, original_data * 2)

        bd1 = minimal_brain_data[0].copy()
        bd2 = minimal_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 *= bd2
        assert np.allclose(bd1.data, original_data * bd2.data)

        bd = minimal_brain_data[0:4].copy()
        c1 = [0.5, 0.5, -0.5, -0.5]
        bd *= c1
        expected = (
            minimal_brain_data[0] * 0.5
            + minimal_brain_data[1] * 0.5
            - minimal_brain_data[2] * 0.5
            - minimal_brain_data[3] * 0.5
        )
        np.testing.assert_almost_equal((bd - expected).sum(), 0, decimal=4)

    def test_inplace_divide(self, minimal_brain_data):
        """Test in-place division with scalars and BrainData."""
        bd = minimal_brain_data[0].copy()
        original_data = bd.data.copy()
        bd /= 2
        assert np.allclose(bd.data, original_data / 2)

        bd1 = minimal_brain_data[0].copy()
        bd2 = minimal_brain_data[0].copy()
        bd2.data = bd2.data + 1  # Avoid division by zero
        original_data = bd1.data.copy()
        bd1 /= bd2
        assert np.allclose(bd1.data, original_data / bd2.data)

    # ==================== Indexing & Concatenation ====================

    @pytest.mark.slow
    def test_indexing(self, minimal_brain_data):
        """Test indexing with lists, ranges, boolean masks, and slices."""
        index = [0, 3, 1]
        assert len(minimal_brain_data[index]) == len(index)
        index = range(4)
        assert len(minimal_brain_data[index]) == len(index)
        # Boolean mask
        bool_idx = np.zeros(len(minimal_brain_data), dtype=bool)
        bool_idx[:5] = True
        assert len(minimal_brain_data[bool_idx]) == 5
        # Slice
        assert len(minimal_brain_data[:3]) == 3
        # Nifti roundtrip
        d = minimal_brain_data.to_nifti()
        assert BrainData(d)

    def test_concatenate(self, minimal_brain_data):
        """Test concatenating BrainData objects from list."""
        out = BrainData(list(minimal_brain_data))
        assert isinstance(out, BrainData)
        assert len(out) == len(minimal_brain_data)

    def test_append(self, minimal_brain_data):
        """Test appending BrainData objects."""
        assert (
            minimal_brain_data.append(minimal_brain_data).shape[0]
            == minimal_brain_data.shape[0] * 2
        )

    # ==================== Statistical Methods ====================

    def test_distance(self, minimal_brain_data):
        """Test distance computation returns Adjacency object."""
        distance = minimal_brain_data.distance(metric="correlation")
        assert isinstance(distance, Adjacency)
        assert distance.n_nodes == minimal_brain_data.shape[0]
