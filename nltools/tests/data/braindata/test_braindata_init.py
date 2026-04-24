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
        assert brain._detected_template.resolution == 3
        assert brain._detected_template.template == "default"

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
        assert brain._detected_template.resolution == 3

    def test_init_resample_true_matched_spaces_no_resample(self):
        """Test that resample=True skips resampling when spaces already match."""

        # Create data in same space as default mask
        mask_img = nib.load(get_brainspace().mask)
        data_same_space = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (10,)), affine=mask_img.affine
        )

        # Should not resample (optimization) and should not show warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(data_same_space, resample=True, verbose=True)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) == 0  # No warning when spaces match

        # Should match original shape
        expected_voxels = mask_img.get_fdata().sum().astype(int)
        assert brain.shape[1] == expected_voxels

    def test_init_mask_template_name_string_2mm_fmriprep(self):
        """Test initialization with template name string: 2mm-MNI152-2009c (fmriprep)."""

        # Create 2mm data
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2
        )

        brain = BrainData(data_2mm, mask="2mm-MNI152-2009c", resample=True)

        # Should use fmriprep 2mm template with exact voxel count
        assert brain.mask is not None
        assert "2mm-MNI152-2009c-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 235840  # Exact voxel count for fmriprep 2mm
        assert brain._detected_template is None  # Explicit mask provided
        assert not brain._mask_was_none

    def test_init_mask_template_name_string_3mm_nilearn(self):
        """Test initialization with template name string: 3mm-MNI152-2009a (nilearn)."""

        # Create 3mm data
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        brain = BrainData(data_3mm, mask="3mm-MNI152-2009a", resample=True)

        # Should use nilearn 3mm template with exact voxel count
        assert brain.mask is not None
        assert "3mm-MNI152-2009a-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 69765  # Exact voxel count for nilearn 3mm
        assert brain._detected_template is None  # Explicit mask provided
        assert not brain._mask_was_none

    @pytest.mark.slow
    def test_init_mask_template_name_string_1mm_nilearn(self):
        """Test initialization with template name string: 1mm-MNI152-2009a (nilearn)."""

        # Create 1mm data
        data_1mm = nib.Nifti1Image(
            np.random.randn(182, 218, 182, 10), affine=np.eye(4) * 1
        )

        brain = BrainData(data_1mm, mask="1mm-MNI152-2009a", resample=True)

        # Should use nilearn 1mm template with exact voxel count
        assert brain.mask is not None
        assert "1mm-MNI152-2009a-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 1886539  # Exact voxel count for nilearn 1mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 1.0, rtol=1e-3)

    def test_init_mask_template_name_string_2mm_fsl(self):
        """Test initialization with template name string: 2mm-MNI152-2009fsl (default)."""

        # Create 2mm data
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2
        )

        brain = BrainData(data_2mm, mask="2mm-MNI152-2009fsl", resample=True)

        # Should use default (fsl) 2mm template with exact voxel count
        assert brain.mask is not None
        assert "2mm-MNI152-2009fsl-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 238955  # Exact voxel count for default 2mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 2.0, rtol=1e-3)

    def test_init_mask_template_name_string_2mm_nilearn(self):
        """Test initialization with template name string: 2mm-MNI152-2009a (nilearn)."""

        # Create 2mm data
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2
        )

        brain = BrainData(data_2mm, mask="2mm-MNI152-2009a", resample=True)

        # Should use nilearn 2mm template with exact voxel count
        assert brain.mask is not None
        assert "2mm-MNI152-2009a-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 235375  # Exact voxel count for nilearn 2mm
        assert brain._detected_template is None  # Explicit mask provided
        assert not brain._mask_was_none

    def test_init_mask_template_name_string_3mm_fsl(self):
        """Test initialization with template name string: 3mm-MNI152-2009fsl (default)."""

        # Create 3mm data
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        brain = BrainData(data_3mm, mask="3mm-MNI152-2009fsl", resample=True)

        # Should use default (fsl) 3mm template with exact voxel count
        assert brain.mask is not None
        assert "3mm-MNI152-2009fsl-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 71020  # Exact voxel count for default 3mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 3.0, rtol=1e-3)

    @pytest.mark.slow
    def test_init_mask_template_name_string_1mm_fmriprep(self):
        """Test initialization with template name string: 1mm-MNI152-2009c (fmriprep)."""

        # Create 1mm data
        data_1mm = nib.Nifti1Image(
            np.random.randn(182, 218, 182, 10), affine=np.eye(4) * 1
        )

        brain = BrainData(data_1mm, mask="1mm-MNI152-2009c", resample=True)

        # Should use fmriprep 1mm template with exact voxel count
        assert brain.mask is not None
        assert "1mm-MNI152-2009c-mask.nii.gz" in brain.mask.get_filename()
        assert brain.shape[1] == 1886574  # Exact voxel count for fmriprep 1mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 1.0, rtol=1e-3)

    def test_init_mask_template_name_string_with_resampling(self):
        """Test template name string with mismatched data resolution (requires resampling)."""

        # Create 3mm data but use 2mm template
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        # With verbose=True, should show resampling warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain = BrainData(
                data_3mm, mask="2mm-MNI152-2009fsl", resample=True, verbose=True
            )
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) > 0  # Warning shown when verbose=True

        # Should be resampled to 2mm template space
        assert brain.shape[1] == 238955  # 2mm voxel count
        assert "2mm-MNI152-2009fsl-mask.nii.gz" in brain.mask.get_filename()

    def test_init_from_brain_data_with_template_name_string(self):
        """Test initialization from BrainData with template name string override."""
        from nltools.templates import get_brainspace

        # Create original BrainData (2mm) using actual MNI152 affine
        default_mask = nib.load(get_brainspace().mask)
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=default_mask.affine
        )
        brain1 = BrainData(data_2mm, mask=None, resample=True)

        # Create new BrainData with template name string override (should resample)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain2 = BrainData(
                brain1, mask="3mm-MNI152-2009a", resample=True, verbose=True
            )
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) > 0  # Warning shown when verbose=True

        # Should be resampled to 3mm nilearn template space
        assert brain2.shape[1] == 69765  # Exact voxel count for nilearn 3mm
        assert "3mm-MNI152-2009a-mask.nii.gz" in brain2.mask.get_filename()
        assert np.allclose(np.abs(brain2.mask.affine[0, 0]), 3.0, rtol=1e-3)

    def test_init_mask_template_name_string_invalid_format(self):
        """Test that invalid template name string format raises error."""

        data = nib.Nifti1Image(np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2)

        # Invalid format - should fall back to file path check
        # If file doesn't exist, should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            BrainData(data, mask="invalid-template-name")

    def test_init_mask_template_name_string_unsupported_resolution(self):
        """Test that unsupported resolution for template raises error."""
        from nltools.templates import resolve_template_name

        # Try to use 3mm with fmriprep (not supported - only 1mm and 2mm)
        with pytest.raises(ValueError, match="Resolution 3mm not supported"):
            resolve_template_name("3mm-MNI152-2009c", file_type="mask")

    def test_init_mask_template_name_string_file_type_brain(self):
        """Test resolve_template_name with file_type='brain'."""
        from nltools.templates import resolve_template_name

        mask_path = resolve_template_name("2mm-MNI152-2009c", file_type="mask")
        brain_path = resolve_template_name("2mm-MNI152-2009c", file_type="brain")

        assert "mask" in mask_path
        assert "brain" in brain_path
        assert mask_path != brain_path

    def test_init_mask_template_name_string_file_type_t1(self):
        """Test resolve_template_name with file_type='T1'."""
        from nltools.templates import resolve_template_name

        mask_path = resolve_template_name("2mm-MNI152-2009c", file_type="mask")
        t1_path = resolve_template_name("2mm-MNI152-2009c", file_type="T1")

        assert "mask" in mask_path
        assert "T1" in t1_path
        assert mask_path != t1_path

    def test_init_mask_template_name_string_invalid_file_type(self):
        """Test that invalid file_type raises error."""
        from nltools.templates import resolve_template_name

        with pytest.raises(ValueError, match="file_type must be"):
            resolve_template_name("2mm-MNI152-2009c", file_type="invalid")

    @pytest.mark.slow
    def test_all_template_voxel_counts(self):
        """Test that all supported templates have correct voxel counts."""

        # Define expected voxel counts for all supported templates
        expected_voxel_counts = {
            "1mm-MNI152-2009a": 1886539,  # nilearn 1mm
            "1mm-MNI152-2009c": 1886574,  # fmriprep 1mm
            "2mm-MNI152-2009fsl": 238955,  # default 2mm
            "2mm-MNI152-2009a": 235375,  # nilearn 2mm
            "2mm-MNI152-2009c": 235840,  # fmriprep 2mm
            "3mm-MNI152-2009fsl": 71020,  # default 3mm
            "3mm-MNI152-2009a": 69765,  # nilearn 3mm
        }
        for template_name, expected_voxels in expected_voxel_counts.items():
            # Extract resolution from template name
            resolution = int(template_name.split("mm")[0])

            # Create data with matching resolution
            if resolution == 1:
                shape = (182, 218, 182)
            elif resolution == 2:
                shape = (91, 109, 91)
            else:  # resolution == 3
                shape = (60, 72, 60)

            data = nib.Nifti1Image(
                np.random.randn(*shape, 10), affine=np.eye(4) * resolution
            )

            # Create BrainData with template name string
            brain = BrainData(data, mask=template_name, resample=True)

            # Verify exact voxel count
            assert brain.shape[1] == expected_voxels, (
                f"Template {template_name} has incorrect voxel count: got {brain.shape[1]}, expected {expected_voxels}"
            )

            # Verify template name is in filename
            assert f"{template_name}-mask.nii.gz" in brain.mask.get_filename()

    def test_all_template_voxel_counts_via_resolve_template_name(self):
        """Test voxel counts via resolve_template_name for all templates."""
        from nltools.templates import resolve_template_name

        # Define expected voxel counts for all supported templates
        expected_voxel_counts = {
            "1mm-MNI152-2009a": 1886539,  # nilearn 1mm
            "1mm-MNI152-2009c": 1886574,  # fmriprep 1mm
            "2mm-MNI152-2009fsl": 238955,  # default 2mm
            "2mm-MNI152-2009a": 235375,  # nilearn 2mm
            "2mm-MNI152-2009c": 235840,  # fmriprep 2mm
            "3mm-MNI152-2009fsl": 71020,  # default 3mm
            "3mm-MNI152-2009a": 69765,  # nilearn 3mm
        }

        # Verify voxel counts by loading masks directly
        for template_name, expected_voxels in expected_voxel_counts.items():
            mask_path = resolve_template_name(template_name, file_type="mask")
            mask = nib.load(mask_path)
            actual_voxels = int(mask.get_fdata().sum())

            assert actual_voxels == expected_voxels, (
                f"Template {template_name} mask has incorrect voxel count: got {actual_voxels}, expected {expected_voxels}"
            )

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
        assert hasattr(brain, "_detected_template")
        assert brain._detected_template.resolution == 2
        assert brain._detected_template.template == "default"

    def test_init_mask_none_auto_detect_3mm(self):
        """Test automatic template detection for 3mm data."""

        # Create 3mm data
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        brain = BrainData(data_3mm, mask=None, resample=True)

        # Should detect and use 3mm template with exact voxel count
        assert brain.shape[1] == 71020  # Exact voxel count for default 3mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 3.0, rtol=1e-3)
        assert brain._detected_template.resolution == 3
        assert brain._detected_template.template == "default"

    @pytest.mark.slow
    def test_init_mask_none_auto_detect_1mm(self):
        """Test automatic template detection for 1mm data (uses nilearn template)."""

        # Create 1mm data (only available in nilearn template)
        data_1mm = nib.Nifti1Image(
            np.random.randn(182, 218, 182, 10), affine=np.eye(4) * 1
        )

        brain = BrainData(data_1mm, mask=None, resample=True)

        # Should detect and use 1mm nilearn template with exact voxel count
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 1.0, rtol=1e-3)
        assert brain.shape[1] == 1886539  # Exact voxel count for nilearn 1mm
        assert brain._detected_template.resolution == 1
        assert brain._detected_template.template == "nilearn"

    def test_init_mask_none_resample_false_exact_match(self):
        """Test auto-detection with resample=False requires exact match."""

        # Create 2mm data that exactly matches template
        mask_2mm = nib.load(get_brainspace().mask)  # Get exact template
        data_2mm = nib.Nifti1Image(
            np.random.randn(*mask_2mm.shape + (10,)), affine=mask_2mm.affine
        )

        brain = BrainData(data_2mm, mask=None, resample=False)

        # Should use template without resampling
        assert brain.shape[1] == mask_2mm.get_fdata().sum().astype(int)

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
        assert brain._detected_template is None or not brain._mask_was_none

    def test_init_mask_none_empty_data(self):
        """Test that empty data with mask=None uses default template."""
        brain = BrainData(data=None, mask=None)

        # Should use default template (2mm default)
        assert brain.mask is not None
        assert hasattr(brain, "_detected_template")
        # Default template info should be None for empty data
        assert brain._detected_template is None or brain._mask_was_none

    def test_init_mask_none_list_consistent_resolution(self, tmpdir):
        """Test auto-detection with list of files (same resolution)."""

        # Create two 3mm files
        data1 = nib.Nifti1Image(np.random.randn(60, 72, 60), affine=np.eye(4) * 3)
        data2 = nib.Nifti1Image(np.random.randn(60, 72, 60), affine=np.eye(4) * 3)

        file1 = str(tmpdir.join("data1.nii.gz"))
        file2 = str(tmpdir.join("data2.nii.gz"))
        data1.to_filename(file1)
        data2.to_filename(file2)

        brain = BrainData([file1, file2], mask=None, resample=True)

        # Should detect 3mm template and use for all files with exact voxel count
        assert brain.shape[0] == 2
        assert brain.shape[1] == 71020  # Exact voxel count for default 3mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 3.0, rtol=1e-3)

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

    def test_init_from_brain_data_with_mask_override(self):
        """Test initialization from BrainData with mask override."""
        from nltools.templates import get_brainspace

        # Create original BrainData (2mm) using actual MNI152 affine
        default_mask = nib.load(get_brainspace().mask)
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=default_mask.affine
        )
        brain1 = BrainData(data_2mm, mask=None, resample=True)

        # Create 3mm mask
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )

        # With verbose=True, should show resampling warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brain2 = BrainData(brain1, mask=mask_3mm, resample=True, verbose=True)
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
            brain2 = BrainData(brain1, mask=mask_3mm, resample=True, verbose=False)
            resample_warnings = [
                warning
                for warning in w
                if "resampling" in str(warning.message).lower()
                and "resample=true" in str(warning.message).lower()
            ]
            assert len(resample_warnings) == 0  # No warning when verbose=False

        # Should be resampled to 3mm mask space
        assert brain2.shape[1] == (60 * 72 * 60)  # All voxels in 3mm space
        assert np.allclose(np.abs(brain2.mask.affine[0, 0]), 3.0, rtol=1e-3)

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
    def test_init_resample_backward_compatibility(self):
        """Test that default behavior (resample=True) matches v0.5.1 behavior."""

        # Create 3mm data (different from default 2mm mask)
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        # Default behavior (resample=True)
        brain_default = BrainData(data_3mm)

        # Explicit resample=True
        brain_explicit = BrainData(data_3mm, resample=True)

        # Should be identical
        assert np.allclose(brain_default.data, brain_explicit.data)
        assert np.allclose(brain_default.mask.affine, brain_explicit.mask.affine)

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

    @pytest.mark.slow
    def test_init_resample_single_vs_multi_image(self):
        """Test resampling works for both single and multi-image data."""

        mask_img = nib.load(get_brainspace().mask)

        # Single image
        data_single = nib.Nifti1Image(
            np.random.randn(*mask_img.shape),
            affine=mask_img.affine * 2,  # Different space
        )
        brain_single = BrainData(data_single, resample=True)
        assert len(brain_single.shape) == 1 or brain_single.shape[0] == 1

        # Multi-image
        data_multi = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (5,)),
            affine=mask_img.affine * 2,  # Different space
        )
        brain_multi = BrainData(data_multi, resample=True)
        assert brain_multi.shape[0] == 5


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

    def test_construct_from_1d_array(self, small_mask):
        mask_img, n_vox, _ = small_mask
        arr = np.arange(n_vox, dtype=np.float32)
        bd = BrainData(arr, mask=mask_img)
        assert bd.shape[-1] == n_vox
        np.testing.assert_array_equal(np.asarray(bd.data).ravel(), arr)

    def test_construct_from_2d_array(self, small_mask):
        mask_img, n_vox, _ = small_mask
        arr = np.random.RandomState(0).randn(4, n_vox).astype(np.float32)
        bd = BrainData(arr, mask=mask_img)
        assert bd.shape == (4, n_vox)
        np.testing.assert_array_equal(np.asarray(bd.data), arr)

    def test_construct_from_array_with_mask_path(self, small_mask):
        _, n_vox, mask_path = small_mask
        arr = np.zeros(n_vox, dtype=np.float32)
        bd = BrainData(arr, mask=mask_path)
        assert bd.shape[-1] == n_vox

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

    def test_array_higher_dim_raises(self, small_mask):
        mask_img, n_vox, _ = small_mask
        with pytest.raises(ValueError, match="must be 1D.*or 2D"):
            BrainData(np.zeros((2, 3, n_vox), dtype=np.float32), mask=mask_img)

    def test_array_roundtrip_through_h5(self, small_mask, tmp_path):
        mask_img, n_vox, _ = small_mask
        arr = np.random.RandomState(1).randn(3, n_vox).astype(np.float32)
        bd = BrainData(arr, mask=mask_img)
        out = tmp_path / "roundtrip.h5"
        bd.write(str(out))
        bd2 = BrainData(str(out))
        assert bd2.shape == bd.shape
        np.testing.assert_array_equal(np.asarray(bd2.data), np.asarray(bd.data))
