"""Contract tests for `BrainData.resample`, `BrainData.apply_mask`, and the
row-metadata policy that both share.

The spec sections under test are "Spatial transformations" and "Row metadata"
in `docs/development/specs/braindata.md`.
"""

import inspect

import nibabel as nib
import numpy as np
import polars as pl
import pytest
from nilearn.image import resample_img, resample_to_img

from nltools.data import BrainData
from nltools.data.braindata.utils import _FIT_STATE_ATTRIBUTES

AFFINE_2MM = np.diag([2.0, 2.0, 2.0, 1.0])
AFFINE_4MM = np.diag([4.0, 4.0, 4.0, 1.0])


def _source_mask():
    """A binary 2 mm mask with a box of support inside an 8x8x8 grid."""
    values = np.zeros((8, 8, 8), dtype=np.uint8)
    values[2:6, 2:6, 2:6] = 1
    return nib.Nifti1Image(values, AFFINE_2MM)


@pytest.fixture
def brain():
    """Three images on the 2 mm grid with row metadata attached."""
    mask = _source_mask()
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((8, 8, 8, 3)).astype(np.float32)
    return BrainData(
        nib.Nifti1Image(vol, AFFINE_2MM),
        mask=mask,
        X=pl.DataFrame({"cond": [1.0, 2.0, 3.0]}),
        Y=pl.DataFrame({"row": [0, 1, 2]}),
        resample=False,
    )


@pytest.fixture
def non_binary_target():
    """A 4 mm target image whose every voxel is non-zero and non-binary.

    If the target's intensities defined the output mask, the result would
    carry all 64 voxels.
    """
    rng = np.random.default_rng(1)
    values = rng.uniform(0.5, 5.0, size=(4, 4, 4)).astype(np.float32)
    return nib.Nifti1Image(values, AFFINE_4MM)


def _expected_mask_on_target(brain, target):
    return resample_to_img(brain.mask, target, interpolation="nearest")


class TestSignatures:
    def test_resample_signature(self):
        parameters = inspect.signature(BrainData.resample).parameters
        assert list(parameters) == ["self", "img", "resolution", "interpolation"]
        for name in ("img", "resolution", "interpolation"):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default is None

    def test_apply_mask_signature(self):
        parameters = inspect.signature(BrainData.apply_mask).parameters
        assert list(parameters) == ["self", "mask"]

    def test_removed_names_are_gone(self):
        assert not hasattr(BrainData, "resample_to")
        from nltools.data.braindata import analysis, io

        assert not hasattr(io, "resample_to")
        assert (
            "resample_mask_to_brain"
            not in inspect.signature(analysis.apply_mask).parameters
        )


class TestResampleValidation:
    def test_requires_exactly_one_target(self, brain):
        with pytest.raises(ValueError, match="both.*img.*and.*resolution"):
            brain.resample(img=_source_mask(), resolution=2.0)
        with pytest.raises(ValueError, match="either.*img.*or.*resolution"):
            brain.resample()

    @pytest.mark.parametrize("resolution", [0.0, -3.0])
    def test_rejects_non_positive_resolution(self, brain, resolution):
        with pytest.raises(ValueError, match="resolution must be positive"):
            brain.resample(resolution=resolution)

    def test_resolution_is_checked_before_any_work(self):
        """The positivity check runs ahead of the work, so an empty object
        reports the bad resolution rather than its own emptiness."""
        with pytest.raises(ValueError, match="resolution must be positive"):
            BrainData().resample(resolution=-1.0)

    def test_rejects_invalid_img_type(self, brain):
        with pytest.raises(TypeError, match="img.*must be"):
            brain.resample(img=123)

    def test_an_empty_object_is_reported_before_the_target_is_opened(self):
        """Argument checks come first, then the object, then any disk access."""
        with pytest.raises(ValueError, match="Cannot resample empty BrainData"):
            BrainData().resample(img="does-not-exist.nii.gz")

    def test_a_resolution_that_erases_the_mask_raises_an_nltools_error(self, brain):
        with pytest.raises(ValueError, match="leaves no voxels"):
            brain.resample(resolution=200.0)


class TestResampleGridSemantics:
    def test_target_image_supplies_only_the_grid(self, brain, non_binary_target):
        result = brain.resample(img=non_binary_target)
        expected = _expected_mask_on_target(brain, non_binary_target)
        support = int(np.count_nonzero(expected.get_fdata() > 0))

        assert 0 < support < 64
        assert result.shape == (3, support)
        np.testing.assert_allclose(result.mask.affine, non_binary_target.affine)
        assert result.mask.shape == non_binary_target.shape

    def test_img_branch_mask_is_the_nearest_resampled_source_mask(
        self, brain, non_binary_target
    ):
        result = brain.resample(img=non_binary_target)
        expected = _expected_mask_on_target(brain, non_binary_target)
        np.testing.assert_array_equal(result.mask.get_fdata(), expected.get_fdata())
        # The target's own intensities are nowhere in the result's mask.
        assert not np.allclose(result.mask.get_fdata(), non_binary_target.get_fdata())

    def test_resolution_branch_mask_is_the_nearest_resampled_source_mask(self, brain):
        result = brain.resample(resolution=4.0)
        target_affine = np.diag([4.0, 4.0, 4.0, 1.0])
        expected = resample_img(
            brain.mask, target_affine=target_affine, interpolation="nearest"
        )
        np.testing.assert_array_equal(result.mask.get_fdata(), expected.get_fdata())
        assert result.shape[1] == int(np.count_nonzero(expected.get_fdata() > 0))

    def test_file_path_target_matches_an_in_memory_target(
        self, brain, non_binary_target, tmp_path
    ):
        path = tmp_path / "target.nii.gz"
        non_binary_target.to_filename(path)
        from_path = brain.resample(img=str(path))
        from_image = brain.resample(img=non_binary_target)
        np.testing.assert_array_equal(from_path.data, from_image.data)
        np.testing.assert_array_equal(
            from_path.mask.get_fdata(), from_image.mask.get_fdata()
        )


class TestResampleResultState:
    @pytest.mark.parametrize("branch", ["resolution", "img"])
    def test_row_metadata_is_preserved(self, brain, non_binary_target, branch):
        kwargs = (
            {"resolution": 4.0}
            if branch == "resolution"
            else {"img": non_binary_target}
        )
        result = brain.resample(**kwargs)
        assert result.X.equals(brain.X)
        assert result.Y.equals(brain.Y)
        assert result.shape[0] == brain.shape[0]

    @pytest.mark.parametrize("branch", ["resolution", "img"])
    def test_fitted_state_is_cleared(self, brain, non_binary_target, branch):
        kwargs = (
            {"resolution": 4.0}
            if branch == "resolution"
            else {"img": non_binary_target}
        )
        brain.fit(model="ridge", X=brain.X.to_numpy(), ridge_alpha=1.0)
        assert any(hasattr(brain, name) for name in _FIT_STATE_ATTRIBUTES)
        result = brain.resample(**kwargs)
        for name in _FIT_STATE_ATTRIBUTES:
            assert not hasattr(result, name)

    def test_result_retains_instance_settings(self, brain):
        brain._h5_compression = "lzf"
        result = brain.resample(resolution=4.0)
        assert result._h5_compression == "lzf"

    def test_result_owns_its_data_mask_and_metadata(self, brain, non_binary_target):
        result = brain.resample(img=non_binary_target)

        assert not np.shares_memory(result.data, brain.data)

        np.asarray(result.mask.dataobj)[:] = 0
        assert np.asarray(brain.mask.dataobj).sum() > 0
        assert np.asarray(non_binary_target.dataobj).sum() > 0

        assert result.X is not brain.X
        result.X = result.X.replace_column(0, pl.Series("cond", [9.0, 9.0, 9.0]))
        assert brain.X["cond"].to_list() == [1.0, 2.0, 3.0]

    def test_source_mask_header_is_not_mutated(self, brain, non_binary_target):
        brain.mask.header.set_sform(brain.mask.affine, code=0)
        non_binary_target.header.set_sform(non_binary_target.affine, code=0)
        brain.resample(img=non_binary_target)
        brain.resample(resolution=4.0)
        assert brain.mask.header.get_sform(coded=True)[1] == 0
        assert non_binary_target.header.get_sform(coded=True)[1] == 0


class TestApplyMask:
    def test_accepts_a_same_grid_mask(self, brain):
        values = np.zeros((8, 8, 8), dtype=np.uint8)
        values[2:4, 2:6, 2:6] = 1
        mask = nib.Nifti1Image(values, AFFINE_2MM)
        result = brain.apply_mask(mask)
        assert result.shape == (3, int(values.sum()))
        assert result.X.equals(brain.X)
        assert result.Y.equals(brain.Y)

    def test_accepts_a_same_grid_braindata_mask(self, brain):
        values = np.zeros((8, 8, 8), dtype=np.uint8)
        values[2:4, 2:6, 2:6] = 1
        mask = BrainData(nib.Nifti1Image(values, AFFINE_2MM), mask=brain.mask)
        result = brain.apply_mask(mask)
        assert result.shape == (3, int(values.sum()))

    def test_rejects_a_mismatched_affine(self, brain):
        mask = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), AFFINE_4MM)
        with pytest.raises(ValueError, match=r"resample\(\)"):
            brain.apply_mask(mask)

    def test_rejects_a_mismatched_shape(self, brain):
        mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), AFFINE_2MM)
        with pytest.raises(ValueError, match=r"resample\(\)"):
            brain.apply_mask(mask)

    def test_same_grid_means_what_the_loader_means_by_it(self, brain):
        """`apply_mask` and the loader share one tolerance for "same grid".

        `check_space_match` accepts a translation drift up to
        ``atol + rtol * |translation|`` with ``rtol=1e-3``. A mask inside that
        band is accepted (and adopts the source affine, so nilearn's stricter
        internal check never sees the drift); one outside it is refused with the
        message that names `resample()`.
        """
        from nltools.data.braindata.io import check_space_match

        values = np.zeros((8, 8, 8), dtype=np.uint8)
        values[2:4, 2:6, 2:6] = 1
        shifted_source = AFFINE_2MM.copy()
        shifted_source[:3, 3] = 100.0
        brain.mask = nib.Nifti1Image(
            np.asarray(brain.mask.dataobj), shifted_source.copy()
        )

        inside = shifted_source.copy()
        inside[0, 3] += 0.05  # tolerance is 1e-8 + 1e-3 * 100 = 0.1 mm
        outside = shifted_source.copy()
        outside[0, 3] += 0.2

        inside_mask = nib.Nifti1Image(values, inside)
        outside_mask = nib.Nifti1Image(values, outside)
        assert check_space_match(inside_mask, brain.mask)
        assert not check_space_match(outside_mask, brain.mask)

        result = brain.apply_mask(inside_mask)
        assert result.shape == (3, int(values.sum()))
        np.testing.assert_array_equal(result.mask.affine, brain.mask.affine)

        with pytest.raises(ValueError, match=r"resample\(\)"):
            brain.apply_mask(outside_mask)

    def test_apply_mask_accepts_a_raw_niimg_on_the_target_grid(self):
        """A raw Niimg mask already on the data's grid is used as given.

        The mask is never re-homed onto the package-default MNI152 template,
        and it is never resampled: `apply_mask` only changes support.
        """
        # Non-MNI space: 4mm isotropic, small grid, offset origin.
        aff = np.diag([4.0, 4.0, 4.0, 1.0])
        aff[:3, 3] = [-20, -20, -20]
        shape = (10, 10, 10)
        rng = np.random.default_rng(0)

        custom_mask = nib.Nifti1Image(np.ones(shape, np.int16), aff)
        data_img = nib.Nifti1Image(rng.standard_normal(shape).astype(np.float32), aff)
        bd = BrainData(data_img, mask=custom_mask)

        # Raw Niimg sub-mask in the SAME non-default space.
        box = np.zeros(shape, np.int16)
        box[2:7, 2:7, 2:7] = 1
        raw_mask = nib.Nifti1Image(box, aff)

        masked = bd.apply_mask(raw_mask)
        assert masked.shape[0] == int(box.sum())

    def test_rejects_an_unsupported_mask_type(self, brain):
        with pytest.raises(TypeError, match="mask must be"):
            brain.apply_mask(123)

    def test_a_mask_wider_than_the_current_support_adds_zero_voxels(self, brain):
        wide = brain.apply_mask(
            nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), AFFINE_2MM)
        )
        assert wide.shape == (3, 512)
        assert int(np.count_nonzero(wide.data[0])) == brain.shape[1]

    def test_rejects_a_multi_volume_mask(self, brain):
        mask = nib.Nifti1Image(np.ones((8, 8, 8, 2), dtype=np.uint8), AFFINE_2MM)
        with pytest.raises(ValueError, match="Mask must be a single image"):
            brain.apply_mask(mask)

    def test_fitted_state_is_cleared(self, brain):
        values = np.zeros((8, 8, 8), dtype=np.uint8)
        values[2:4, 2:6, 2:6] = 1
        brain.fit(model="ridge", X=brain.X.to_numpy(), ridge_alpha=1.0)
        assert any(hasattr(brain, name) for name in _FIT_STATE_ATTRIBUTES)
        result = brain.apply_mask(nib.Nifti1Image(values, AFFINE_2MM))
        for name in _FIT_STATE_ATTRIBUTES:
            assert not hasattr(result, name)

    def test_result_owns_its_mask_and_metadata(self, brain):
        values = np.zeros((8, 8, 8), dtype=np.uint8)
        values[2:4, 2:6, 2:6] = 1
        mask = nib.Nifti1Image(values, AFFINE_2MM)
        result = brain.apply_mask(mask)

        np.asarray(result.mask.dataobj)[:] = 0
        assert np.asarray(mask.dataobj).sum() > 0

        assert result.X is not brain.X
        result.X = result.X.replace_column(0, pl.Series("cond", [7.0, 7.0, 7.0]))
        assert brain.X["cond"].to_list() == [1.0, 2.0, 3.0]


class TestRowMetadataCategories:
    """Row metadata follows the output's leading axis, category by category."""

    @pytest.mark.parametrize(
        "category",
        ["value_transform", "arithmetic", "smoothing", "masking", "resampling"],
    )
    def test_row_preserving_categories_retain_metadata(self, brain, category):
        results = {
            "value_transform": lambda: brain.scale(),
            "arithmetic": lambda: brain + 2,
            "smoothing": lambda: brain.smooth(4),
            "masking": lambda: brain.apply_mask(_source_mask()),
            "resampling": lambda: brain.resample(resolution=4.0),
        }
        result = results[category]()
        assert result.X.equals(brain.X)
        assert result.Y.equals(brain.Y)
        assert result.shape[0] == brain.shape[0]

    def test_selection_applies_the_same_ordering_to_metadata(self, brain):
        result = brain[[2, 0]]
        assert result.X["cond"].to_list() == [3.0, 1.0]
        assert result.Y["row"].to_list() == [2, 0]

    def test_reductions_clear_metadata(self, brain):
        result = brain.mean()
        assert result.X.is_empty()
        assert result.Y.is_empty()

    def test_pairwise_transformation_clears_x_and_installs_its_own_y(self, brain):
        brain.Y = pl.DataFrame({"label": [0, 1, 0], "group": [1, 1, 1]})
        result = brain.transform_pairwise()
        assert result.X.is_empty()
        assert not result.Y.is_empty()
        assert result.Y.height == result.shape[0]
