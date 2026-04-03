import numpy as np
import pytest

from nltools.data import BrainData, Adjacency


shape_3d = (91, 109, 91)
shape_2d = (6, 238955)


class TestBrainDataCore:
    """Test BrainData dunders, properties, arithmetic, stats, indexing, append, copy, and distance."""

    # ==================== Properties & Statistics ====================

    def test_shape(self, sim_brain_data):
        """Test shape property returns correct dimensions."""
        assert sim_brain_data.shape == shape_2d

    def test_mean(self, sim_brain_data):
        """Test mean computation across different axes."""
        assert sim_brain_data.mean().shape[0] == shape_2d[1]
        assert sim_brain_data.mean().shape[0] == shape_2d[1]
        assert len(sim_brain_data.mean(axis=1)) == shape_2d[0]
        with pytest.raises(ValueError):
            sim_brain_data.mean(axis="1")
        assert isinstance(sim_brain_data[0].mean(), (float, np.floating))

    def test_median(self, sim_brain_data):
        """Test median computation across different axes."""
        assert sim_brain_data.median().shape[0] == shape_2d[1]
        assert sim_brain_data.median().shape[0] == shape_2d[1]
        assert len(sim_brain_data.median(axis=1)) == shape_2d[0]
        with pytest.raises(ValueError):
            sim_brain_data.median(axis="1")
        assert isinstance(sim_brain_data[0].median(), (float, np.floating))

    def test_std(self, sim_brain_data):
        """Test standard deviation computation."""
        assert sim_brain_data.std().shape[0] == shape_2d[1]

    def test_sum(self, sim_brain_data):
        """Test sum aggregation."""
        s = sim_brain_data.sum()
        assert s.shape == sim_brain_data[1].shape

    # ==================== Arithmetic Operations ====================

    def test_add(self, sim_brain_data):
        """Test addition of BrainData objects and scalars."""
        new = sim_brain_data + sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (value + sim_brain_data[0]).mean() == (sim_brain_data[0] + value).mean()

    def test_subtract(self, sim_brain_data):
        """Test subtraction of BrainData objects and scalars."""
        new = sim_brain_data - sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (-value - (-1) * sim_brain_data[0]).mean() == (
            sim_brain_data[0] - value
        ).mean()

    def test_multiply(self, sim_brain_data):
        """Test multiplication of BrainData objects, scalars, and arrays."""
        new = sim_brain_data * sim_brain_data
        assert new.shape == shape_2d
        value = 10
        assert (value * sim_brain_data[0]).mean() == (sim_brain_data[0] * value).mean()
        c1 = [0.5, 0.5, -0.5, -0.5]
        new = sim_brain_data[0:4] * c1
        new2 = (
            sim_brain_data[0] * 0.5
            + sim_brain_data[1] * 0.5
            - sim_brain_data[2] * 0.5
            - sim_brain_data[3] * 0.5
        )
        np.testing.assert_almost_equal((new - new2).sum(), 0, decimal=4)

    def test_divide(self, sim_brain_data):
        """Test division of BrainData objects and scalars."""
        new = sim_brain_data / sim_brain_data
        assert new.shape == shape_2d
        np.testing.assert_almost_equal(new.mean(axis=0).mean(), 1, decimal=6)
        value = 10
        new2 = sim_brain_data / value
        np.testing.assert_almost_equal(
            ((new2 * value) - new2).mean().mean(), 0, decimal=2
        )

    def test_inplace_add(self, sim_brain_data):
        """Test in-place addition with scalars and BrainData."""
        # Test in-place add with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd += 5
        assert np.allclose(bd.data, original_data + 5)

        # Test in-place add with BrainData
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 += bd2
        assert np.allclose(bd1.data, original_data + bd2.data)

    def test_inplace_subtract(self, sim_brain_data):
        """Test in-place subtraction with scalars and BrainData."""
        # Test in-place subtract with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd -= 3
        assert np.allclose(bd.data, original_data - 3)

        # Test in-place subtract with BrainData
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 -= bd2
        assert np.allclose(bd1.data, original_data - bd2.data)

    def test_inplace_multiply(self, sim_brain_data):
        """Test in-place multiplication with scalars, BrainData, and arrays."""
        # Test in-place multiply with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd *= 2
        assert np.allclose(bd.data, original_data * 2)

        # Test in-place multiply with BrainData
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        original_data = bd1.data.copy()
        bd1 *= bd2
        assert np.allclose(bd1.data, original_data * bd2.data)

        # Test in-place multiply with array
        bd = sim_brain_data[0:4].copy()
        c1 = [0.5, 0.5, -0.5, -0.5]
        bd *= c1
        expected = (
            sim_brain_data[0] * 0.5
            + sim_brain_data[1] * 0.5
            - sim_brain_data[2] * 0.5
            - sim_brain_data[3] * 0.5
        )
        np.testing.assert_almost_equal((bd - expected).sum(), 0, decimal=4)

    def test_inplace_divide(self, sim_brain_data):
        """Test in-place division with scalars and BrainData."""
        # Test in-place divide with scalar
        bd = sim_brain_data[0].copy()
        original_data = bd.data.copy()
        bd /= 2
        assert np.allclose(bd.data, original_data / 2)

        # Test in-place divide with BrainData
        bd1 = sim_brain_data[0].copy()
        bd2 = sim_brain_data[0].copy()
        bd2.data = bd2.data + 1  # Avoid division by zero
        original_data = bd1.data.copy()
        bd1 /= bd2
        assert np.allclose(bd1.data, original_data / bd2.data)

    # ==================== Indexing & Concatenation ====================

    def test_indexing(self, sim_brain_data):
        """Test indexing with lists, ranges, boolean masks, and slices."""
        index = [0, 3, 1]
        assert len(sim_brain_data[index]) == len(index)
        index = range(4)
        assert len(sim_brain_data[index]) == len(index)
        index = sim_brain_data.Y == 1
        assert len(sim_brain_data[index.values.flatten()]) == index.values.sum()
        assert len(sim_brain_data[index]) == index.values.sum()
        assert len(sim_brain_data[:3]) == 3
        d = sim_brain_data.to_nifti()
        assert d.shape[0:3] == shape_3d
        assert BrainData(d)

    def test_concatenate(self, sim_brain_data):
        """Test concatenating BrainData objects from list."""
        out = BrainData([x for x in sim_brain_data])
        assert isinstance(out, BrainData)
        assert len(out) == len(sim_brain_data)

    def test_append(self, sim_brain_data):
        """Test appending BrainData objects."""
        assert sim_brain_data.append(sim_brain_data).shape[0] == shape_2d[0] * 2

    # ==================== Statistical Methods ====================

    def test_distance(self, sim_brain_data):
        """Test distance computation returns Adjacency object."""
        distance = sim_brain_data.distance(metric="correlation")
        assert isinstance(distance, Adjacency)
        assert distance.n_nodes == shape_2d[0]
