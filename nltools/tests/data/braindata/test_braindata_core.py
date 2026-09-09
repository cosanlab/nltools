import numpy as np
import polars as pl
import pytest

from nltools.data import BrainData, Adjacency


class TestBrainDataCore:
    """Test BrainData dunders, properties, arithmetic, stats, indexing, append, copy, and distance."""

    # ==================== Properties & Statistics ====================

    def test_shape(self, minimal_brain_data):
        """Test shape property returns correct dimensions."""
        assert minimal_brain_data.shape == (50, 5)

    def test_equality_compares_in_memory_mask_affines(self):
        import nibabel as nib

        mask_data = np.ones((2, 2, 2), dtype=np.uint8)
        mask_a = nib.Nifti1Image(mask_data, np.eye(4))
        mask_b = nib.Nifti1Image(mask_data, np.diag([2.0, 2.0, 2.0, 1.0]))
        data = np.zeros((1, mask_data.size))

        assert BrainData(data, mask=mask_a) != BrainData(data, mask=mask_b)

    def test_equality_accepts_equivalent_in_memory_masks(self):
        import nibabel as nib

        mask_data = np.ones((2, 2, 2), dtype=np.uint8)
        mask_a = nib.Nifti1Image(mask_data, np.eye(4))
        mask_b = nib.Nifti1Image(mask_data.copy(), np.eye(4))
        data = np.zeros((1, mask_data.size))

        assert BrainData(data, mask=mask_a) == BrainData(data.copy(), mask=mask_b)

    def test_equality_compares_in_memory_mask_voxels(self):
        import nibabel as nib

        mask_a_data = np.zeros((2, 2, 2), dtype=np.uint8)
        mask_b_data = np.zeros((2, 2, 2), dtype=np.uint8)
        mask_a_data.flat[[0, 1]] = 1
        mask_b_data.flat[[0, 2]] = 1
        mask_a = nib.Nifti1Image(mask_a_data, np.eye(4))
        mask_b = nib.Nifti1Image(mask_b_data, np.eye(4))
        data = np.zeros((1, 2))

        assert BrainData(data, mask=mask_a) != BrainData(data, mask=mask_b)

    def test_copy_owns_complete_fitted_state(self, minimal_brain_data):
        """Copying a fitted BrainData produces an independent snapshot."""
        X = np.random.default_rng(0).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)

        copied = minimal_brain_data.copy()

        copied.data[0, 0] = 11.0
        copied.X_[0, 0] = 12.0
        copied.model_.coef_[0, 0] = 13.0
        copied.ridge_weights.data[0, 0] = 14.0
        copied.mask.get_fdata(caching="fill")[0, 0, 0] = 0.0

        assert minimal_brain_data.data[0, 0] != 11.0
        assert minimal_brain_data.X_[0, 0] != 12.0
        assert minimal_brain_data.model_.coef_[0, 0] != 13.0
        assert minimal_brain_data.ridge_weights.data[0, 0] != 14.0
        assert minimal_brain_data.mask.get_fdata()[0, 0, 0] != 0.0

    def test_create_empty_drops_fitted_state(self, minimal_brain_data):
        X = np.random.default_rng(1).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)

        empty = minimal_brain_data.create_empty()

        assert empty.data.size == 0
        assert not hasattr(empty, "model_")
        assert not hasattr(empty, "X_")
        assert not hasattr(empty, "ridge_weights")

    def test_inplace_arithmetic_drops_fitted_state(self, minimal_brain_data):
        X = np.random.default_rng(2).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)

        minimal_brain_data += 1.0

        assert not hasattr(minimal_brain_data, "model_")
        assert not hasattr(minimal_brain_data, "X_")
        assert not hasattr(minimal_brain_data, "ridge_weights")

    def test_setitem_drops_fitted_state(self, minimal_brain_data):
        replacement = minimal_brain_data[0]
        X = np.random.default_rng(3).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)

        minimal_brain_data[0] = replacement

        assert not hasattr(minimal_brain_data, "model_")
        assert not hasattr(minimal_brain_data, "X_")
        assert not hasattr(minimal_brain_data, "ridge_weights")

    def test_failed_setitem_preserves_data_and_fitted_state(self, minimal_brain_data):
        replacement = minimal_brain_data[0]
        replacement.X = pl.DataFrame({"unexpected": [1.0]})
        X = np.random.default_rng(4).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)
        original_data = minimal_brain_data.data.copy()
        original_model = minimal_brain_data.model_
        original_weights = minimal_brain_data.ridge_weights

        with pytest.raises(ValueError, match="self.X is the same size"):
            minimal_brain_data[0] = replacement

        np.testing.assert_array_equal(minimal_brain_data.data, original_data)
        assert minimal_brain_data.model_ is original_model
        assert minimal_brain_data.ridge_weights is original_weights

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
