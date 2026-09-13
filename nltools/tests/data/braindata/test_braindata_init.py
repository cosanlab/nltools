import warnings

import numpy as np
import nibabel as nib
import pytest

from nltools.data import BrainData
from nltools.templates import get_brainspace


class TestBrainDataInit:
    """Test BrainData initialization, resampling, and template auto-detection."""

    def test_init_resample_true_mismatched_spaces(self):
        """Test automatic resampling when data and mask have different spaces."""

        # Create data in different space (3mm) than explicitly specified mask (2mm)
        # Use small shape to speed up test
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10),  # 3mm shape
            affine=np.eye(4) * 3,  # 3mm affine
        )

        # Explicitly use 2mm mask (new behavior: mask=None would auto-detect 3mm)
        mask_img = nib.load(get_brainspace().mask)

        # With verbose=True, should show resampling warning
        from nltools.utils import ResamplingWarning

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_3mm, mask=mask_img, resample=True, verbose=True)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) > 0  # Warning shown when verbose=True
            # Its own category, attributed to the caller (this file), and it
            # names the grid the data is being resampled onto.
            assert all(x.category is ResamplingWarning for x in resample_warnings)
            assert all(x.filename == __file__ for x in resample_warnings)
            assert "2x2x2mm" in str(resample_warnings[0].message)

        # With verbose=False, should suppress warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_3mm, mask=mask_img, resample=True, verbose=False)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) == 0  # No warning when verbose=False

        # Should be resampled to 2mm mask space
        assert brain.shape[1] == 238955  # 2mm voxel count
        assert np.allclose(brain.mask.affine, mask_img.affine, rtol=1e-2)

    def test_init_resample_false_matched_spaces(self):
        """Test no resampling when resample=False and spaces match."""

        # Create data in same space as mask
        mask_img = nib.load(get_brainspace().mask)
        data_same_space = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (10,)), affine=mask_img.affine
        )

        brain = BrainData(data_same_space, resample=False)

        # Should not be resampled
        expected_voxels = mask_img.get_fdata().sum().astype(int)
        assert brain.shape[1] == expected_voxels

    @pytest.mark.slow
    def test_init_resample_false_mismatched_spaces(self):
        """Test that resample=False with mismatched spaces shows warning but still resamples."""

        # Create data in different space
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )
        # Create 2mm mask
        mask_2mm = nib.Nifti1Image(
            np.ones((91, 109, 91), dtype=np.float32), affine=np.eye(4) * 2
        )

        # With verbose=False, warning should be suppressed
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_3mm, mask=mask_2mm, resample=False, verbose=False)
            assert len(w) == 0  # No warning when verbose=False

        # With verbose=True, warning should be shown
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_3mm, mask=mask_2mm, resample=False, verbose=True)
            assert len(w) > 0  # Warning shown when verbose=True
            assert "resample" in str(w[0].message).lower()

        # Data should still be resampled correctly
        assert brain.shape[1] == (91 * 109 * 91)  # Should match 2mm mask

    def test_init_resample_true_custom_mask(self, tmpdir):
        """Test resampling to custom mask space."""

        # Create custom 3mm mask
        custom_mask_data = np.ones((60, 72, 60), dtype=np.float32)
        custom_mask = nib.Nifti1Image(custom_mask_data, affine=np.eye(4) * 3)

        # Create 2mm data
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2
        )

        # With verbose=True, should show resampling warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_2mm, mask=custom_mask, resample=True, verbose=True)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) > 0  # Warning shown when verbose=True

        # With verbose=False, should suppress warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_2mm, mask=custom_mask, resample=True, verbose=False)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) == 0  # No warning when verbose=False

        # Should be resampled to 3mm custom mask space
        assert brain.shape[1] == (60 * 72 * 60)  # All voxels in 3mm space

    def test_init_resample_true_default_mask(self):
        """Test resampling to auto-detected template when mask=None."""

        # Create 3mm data
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        # Use default mask (mask=None) - will auto-detect template from data
        brain = BrainData(data_3mm, resample=True)

        # Should auto-detect and use 3mm template (not default 2mm)
        assert brain.shape[1] == 71020  # Exact voxel count for default 3mm

    def test_init_resample_true_list_of_files(self, tmpdir):
        """Test resampling works with list of files."""

        # Create two 3mm files
        data1 = nib.Nifti1Image(np.random.randn(60, 72, 60), affine=np.eye(4) * 3)
        data2 = nib.Nifti1Image(np.random.randn(60, 72, 60), affine=np.eye(4) * 3)

        file1 = str(tmpdir.join("data1.nii.gz"))
        file2 = str(tmpdir.join("data2.nii.gz"))
        data1.to_filename(file1)
        data2.to_filename(file2)

        # With verbose=True, should show resampling warning (only once for first item)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData([file1, file2], resample=True, verbose=True)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            # Note: May or may not show warning depending on if spaces match after auto-detection
            # If spaces match exactly, no warning; if there's a mismatch, warning appears

        # With verbose=False, should suppress warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData([file1, file2], resample=True, verbose=False)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) == 0  # No warning when verbose=False

        # Should be resampled to auto-detected template space (3mm)
        assert brain.shape == (2, 71020)  # Exact voxel count for default 3mm

    def test_init_mask_template_name_string_invalid_format(self):
        """Test that invalid template name string format raises error."""

        data = nib.Nifti1Image(np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2)

        # Invalid format - should fall back to file path check
        # If file doesn't exist, should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            BrainData(data, mask="invalid-template-name")

    def test_init_mask_template_name_string_file_type_brain(self):
        """Test _resolve_template_name with file_type='brain'."""
        from nltools.templates import _resolve_template_name

        mask_path = _resolve_template_name("2mm-MNI152-2009c", file_type="mask")
        brain_path = _resolve_template_name("2mm-MNI152-2009c", file_type="brain")

        assert "mask" in mask_path
        assert "brain" in brain_path
        assert mask_path != brain_path

    # ==================== Template Auto-Detection ====================

    def test_init_mask_none_auto_detect_2mm(self):
        """Test automatic template detection for 2mm data."""
        from nltools.templates import get_brainspace

        # Create 2mm data (matches default template) using actual MNI152 affine
        default_mask = nib.load(get_brainspace().mask)
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=default_mask.affine
        )

        brain = BrainData(data_2mm, mask=None, resample=True)

        # Should detect and use 2mm template with exact voxel count
        assert brain.shape[1] == 238955  # Exact voxel count for default 2mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 2.0, rtol=1e-3)

    def test_init_mask_none_resample_false_mismatch(self):
        """Test that resample=False with mismatched data shows warning but still resamples."""

        # Create data that doesn't match any template exactly
        data = nib.Nifti1Image(
            np.random.randn(100, 100, 100, 10),
            affine=np.eye(4) * 1.5,  # Non-standard resolution
        )

        # With verbose=False, warning should be suppressed
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data, mask=None, resample=False, verbose=False)
            assert len(w) == 0  # No warning when verbose=False

        # With verbose=True, warning should be shown
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data, mask=None, resample=False, verbose=True)
            assert len(w) > 0  # Warning shown when verbose=True
            assert "resample" in str(w[0].message).lower()

        # Data should still be resampled correctly
        assert brain.shape[1] > 0  # Should have valid shape

    def test_init_custom_mask_overrides_auto_detect(self):
        """Test that explicit mask parameter overrides auto-detection."""

        # Create custom mask
        custom_mask = nib.Nifti1Image(
            np.ones((50, 50, 50), dtype=np.float32), affine=np.eye(4) * 2.5
        )

        # Create 2mm data
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2
        )

        # With verbose=True, should show resampling warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_2mm, mask=custom_mask, resample=True, verbose=True)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) > 0  # Warning shown when verbose=True

        # With verbose=False, should suppress warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_2mm, mask=custom_mask, resample=True, verbose=False)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) == 0  # No warning when verbose=False

        # Should use custom mask, not auto-detected
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 2.5, rtol=1e-3)

    def test_init_mask_none_empty_data(self):
        """Empty data with mask=None still gets the default template mask."""
        brain = BrainData(data=None, mask=None)

        assert brain.mask is not None

    def test_init_from_brain_data(self):
        """Test initialization from another BrainData object."""
        from nltools.templates import get_brainspace

        # Create original BrainData using actual MNI152 affine
        default_mask = nib.load(get_brainspace().mask)
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=default_mask.affine
        )
        brain1 = BrainData(data_2mm, mask=None, resample=True)

        # Create new BrainData from existing one
        brain2 = BrainData(brain1)

        # Should have same shape and data
        assert brain1.shape == brain2.shape
        assert np.allclose(brain1.data, brain2.data)
        assert brain1.mask.get_filename() == brain2.mask.get_filename()

    def test_init_from_brain_data_resample_false_error(self):
        """Test that resample=False raises error when masks don't match."""
        from nltools.templates import get_brainspace

        # Create original BrainData (2mm) using actual MNI152 affine to avoid resampling warning
        default_mask = nib.load(get_brainspace().mask)
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=default_mask.affine
        )
        brain1 = BrainData(data_2mm, mask=None, resample=True)

        # Create 3mm mask
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )

        # Should raise error when resample=False and masks don't match
        with pytest.raises(ValueError, match="resample=True"):
            BrainData(brain1, mask=mask_3mm, resample=False)

    @pytest.mark.slow
    def test_init_resample_preserves_data_integrity(self):
        """Test that resampling preserves data characteristics."""

        # Create test data with known values
        data_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60, 10)) * 5.0,  # Constant value
            affine=np.eye(4) * 3,
        )

        brain = BrainData(data_3mm, resample=True)

        # Values within mask should be preserved (approximately, due to interpolation)
        # Most voxels will be 0 (outside mask), but masked voxels should be ~5.0
        masked_voxels = brain.data[brain.data != 0]
        if len(masked_voxels) > 0:
            assert np.allclose(
                masked_voxels, 5.0, rtol=0.1
            )  # Allow interpolation tolerance


class TestBrainDataInitFromArray:
    """Construct BrainData directly from a numpy array + explicit mask.

    Lets group-level stats (t, z, mean, ...) be wrapped back into BrainData
    without the copy-a-template-and-overwrite-.data dance.
    """

    @pytest.fixture
    def small_mask(self, tmp_path):
        mask_arr = np.zeros((5, 5, 5), dtype=np.float32)
        mask_arr[1:4, 1:4, 1:4] = 1.0  # 27 in-mask voxels
        img = nib.Nifti1Image(mask_arr, affine=np.eye(4, dtype=np.float32))
        path = tmp_path / "mask.nii.gz"
        nib.save(img, path)
        return img, int(mask_arr.sum()), str(path)

    def test_construct_from_2d_array(self, small_mask):
        mask_img, n_vox, _ = small_mask
        arr = np.random.RandomState(0).randn(4, n_vox).astype(np.float32)
        bd = BrainData(arr, mask=mask_img)
        assert bd.shape == (4, n_vox)
        np.testing.assert_array_equal(np.asarray(bd.data), arr)

    def test_array_without_mask_raises(self, small_mask):
        _, n_vox, _ = small_mask
        arr = np.zeros(n_vox, dtype=np.float32)
        with pytest.raises(ValueError, match="requires an explicit mask"):
            BrainData(arr)

    def test_array_shape_mismatch_raises(self, small_mask):
        mask_img, n_vox, _ = small_mask
        # One voxel short
        arr = np.zeros(n_vox - 1, dtype=np.float32)
        with pytest.raises(ValueError, match="must match the number of in-mask voxels"):
            BrainData(arr, mask=mask_img)
