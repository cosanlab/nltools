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

    def test_mean_reduces_across_images(self, minimal_brain_data):
        """`mean()` reduces along the image axis and returns a BrainData."""
        out = minimal_brain_data.mean()
        assert isinstance(out, BrainData)
        np.testing.assert_allclose(out.data, minimal_brain_data.data.mean(axis=0))

    def test_repr_with_unnamed_in_memory_mask(self, minimal_brain_data):
        """#447: an in-memory mask has no filename, so the repr reads `mask=None`."""
        assert minimal_brain_data.mask.get_filename() is None
        assert "mask=None" in repr(minimal_brain_data)

    def test_equality_accepts_equivalent_in_memory_masks(self):
        import nibabel as nib

        mask_data = np.ones((2, 2, 2), dtype=np.uint8)
        mask_a = nib.Nifti1Image(mask_data, np.eye(4))
        mask_b = nib.Nifti1Image(mask_data.copy(), np.eye(4))
        data = np.zeros((1, mask_data.size))

        assert BrainData(data, mask=mask_a) == BrainData(data.copy(), mask=mask_b)

    def test_equality_compares_shapes_too(self):
        """Unequal shapes are unequal, and never raise out of `__eq__`.

        `np.all(a == b)` broadcasts, so one all-ones map read as equal to a
        stack of three identical ones, and two rows against three raised.
        """
        import nibabel as nib

        mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
        single = BrainData(np.ones(8), mask=mask)
        three = BrainData(np.ones((3, 8)), mask=mask)
        two = BrainData(np.ones((2, 8)), mask=mask)

        assert single != three
        assert two != three

    def test_copy_owns_complete_fitted_state(self, minimal_brain_data):
        """Copying a fitted BrainData produces an independent snapshot."""
        X = np.random.default_rng(0).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)

        copied = minimal_brain_data.copy()

        copied.data[0, 0] = 11.0
        copied.model._estimator.coef_[0, 0] = 13.0
        copied.model.betas.data[0, 0] = 14.0
        copied.mask.get_fdata(caching="fill")[0, 0, 0] = 0.0

        assert minimal_brain_data.data[0, 0] != 11.0
        assert minimal_brain_data.model._estimator.coef_[0, 0] != 13.0
        assert minimal_brain_data.model.betas.data[0, 0] != 14.0
        assert minimal_brain_data.mask.get_fdata()[0, 0, 0] != 0.0

    def test_create_empty_drops_fitted_state(self, minimal_brain_data):
        X = np.random.default_rng(1).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)

        empty = minimal_brain_data.create_empty()

        assert empty.data.size == 0
        assert empty.model is None
        assert not hasattr(empty, "X_")

    def test_inplace_arithmetic_drops_fitted_state(self, minimal_brain_data):
        X = np.random.default_rng(2).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)

        minimal_brain_data += 1.0

        assert minimal_brain_data.model is None
        assert not hasattr(minimal_brain_data, "X_")

    def test_setitem_drops_fitted_state(self, minimal_brain_data):
        replacement = minimal_brain_data[0]
        X = np.random.default_rng(3).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)

        minimal_brain_data[0] = replacement

        assert minimal_brain_data.model is None
        assert not hasattr(minimal_brain_data, "X_")

    def test_failed_setitem_preserves_data_and_fitted_state(self, minimal_brain_data):
        replacement = minimal_brain_data[0]
        replacement.X = pl.DataFrame({"unexpected": [1.0]})
        X = np.random.default_rng(4).standard_normal((len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)
        original_data = minimal_brain_data.data.copy()
        original_fit = minimal_brain_data.model

        with pytest.raises(ValueError, match="self.X is the same size"):
            minimal_brain_data[0] = replacement

        np.testing.assert_array_equal(minimal_brain_data.data, original_data)
        assert minimal_brain_data.model is original_fit

    @pytest.mark.parametrize("method", ["median"])
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

    # ==================== Arithmetic Operations ====================

    def test_add(self, minimal_brain_data):
        """Test addition of BrainData objects and scalars."""
        new = minimal_brain_data + minimal_brain_data
        assert new.shape == minimal_brain_data.shape
        value = 10
        assert (value + minimal_brain_data[0]).mean() == (
            minimal_brain_data[0] + value
        ).mean()

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

    def test_append_rejects_unknown_keyword(self, minimal_brain_data):
        """An unexpected keyword raises instead of being silently swallowed."""
        with pytest.raises(TypeError):
            minimal_brain_data.append(minimal_brain_data, foo=1)

    def test_single_image_is_one_observation(self, minimal_brain_data):
        """A single image is one image, not one observation per voxel.

        `__len__` and `__getitem__` both count along the image axis, which for a
        1-D single image is a length of one — not the voxel count, which made
        `len()` meaningless and iteration impossible.
        """
        single = minimal_brain_data[0]
        assert single.shape == (5,)
        assert len(single) == 1
        assert len(list(single)) == 1
        np.testing.assert_array_equal(single[0].data, single.data)
        with pytest.raises(IndexError):
            single[1]

    def test_empty_data_has_length_zero(self, minimal_brain_data):
        assert len(minimal_brain_data.create_empty()) == 0

    def test_setitem_replaces_metadata_by_column_name(self):
        """Row assignment matches metadata columns by name, and keeps their dtypes.

        Routing the replacement through one NumPy matrix matched columns by
        position — so a differently ordered replacement installed the wrong
        values — and collapsed a mixed-dtype frame to Object.
        """
        import nibabel as nib

        mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
        brain = BrainData(
            np.zeros((3, 8)),
            mask=mask,
            X=pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        )
        replacement = BrainData(
            np.ones((1, 8)),
            mask=mask,
            X=pl.DataFrame({"b": ["q"], "a": [9]}),
        )

        brain[0] = replacement

        assert brain.X.schema == pl.Schema([("a", pl.Int64), ("b", pl.String)])
        assert brain.X.to_dicts()[0] == {"a": 9, "b": "q"}

    def test_setitem_writes_a_single_column_metadata_frame(self):
        """A one-column metadata frame is the common case and must not crash.

        Polars hands back a read-only zero-copy view for a single numeric
        column, so assigning into `to_numpy()` raised before anything was
        written.
        """
        import nibabel as nib

        mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
        brain = BrainData(
            np.zeros((3, 8)), mask=mask, Y=pl.DataFrame({"label": [1.0, 2.0, 3.0]})
        )
        replacement = BrainData(
            np.ones((1, 8)), mask=mask, Y=pl.DataFrame({"label": [9.0]})
        )

        brain[0] = replacement

        assert brain.Y["label"].to_list() == [9.0, 2.0, 3.0]
        np.testing.assert_array_equal(brain.data[0], replacement.data)
        assert brain.model is None

    def test_setitem_rejects_unknown_metadata_columns(self):
        import nibabel as nib

        mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
        brain = BrainData(
            np.zeros((3, 8)), mask=mask, Y=pl.DataFrame({"label": [1.0, 2.0, 3.0]})
        )
        replacement = BrainData(
            np.ones((1, 8)), mask=mask, Y=pl.DataFrame({"other": [9.0]})
        )

        with pytest.raises(ValueError, match="compatible Y metadata"):
            brain[0] = replacement
        assert brain.Y["label"].to_list() == [1.0, 2.0, 3.0]

    # ==================== Statistical Methods ====================

    def test_distance(self, minimal_brain_data):
        """Test distance computation returns Adjacency object."""
        distance = minimal_brain_data.distance(metric="correlation")
        assert isinstance(distance, Adjacency)
        assert distance.n_nodes == minimal_brain_data.shape[0]
