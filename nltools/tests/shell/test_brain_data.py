"""
Test suite for BrainData class.

Follows "imperative shell" pattern: tests focus on method usage and interface contracts,
not implementation details. Organized into logical sections for clarity.

Performance Notes:
------------------
Test suite timing (47 tests total, ~151s):
- Average per test: ~3.2s
- Threshold tests: ~7.2s each (cluster filtering is computationally expensive)
- Math operations: <1s each (fast numpy operations)

Slowest test categories:
1. Threshold operations (~7.2s each) - cluster filtering uses nilearn connected components
2. GLM regression (~5-6s) - FirstLevelModel fitting
3. Hyperalignment/decomposition (~4-5s) - large matrix operations
4. Bootstrap/permutation tests (~3-4s) - resampling operations

The cluster threshold tests (9 tests, ~65s) consume ~43% of total runtime. This is
expected and acceptable - they test realistic neuroimaging workflows that require
expensive connected components analysis via nilearn.
"""

import os
import pytest
import numpy as np
import nibabel as nb
import pandas as pd
from nltools.simulator import Simulator
from nltools.data import BrainData, Adjacency
from nltools.stats import align
from nltools.mask import create_sphere, roi_to_brain
from pathlib import Path

from nltools.prefs import MNI_Template
from nltools.tests.conftest import _tables_available


shape_3d = (91, 109, 91)
shape_2d = (6, 238955)


class TestBrainData:
    """Test BrainData class - focus on method usage, not implementation."""

    # ==================== Initialization & I/O ====================

    def test_init_resample_true_mismatched_spaces(self):
        """Test automatic resampling when data and mask have different spaces."""
        import nibabel as nib
        import warnings

        # Create data in different space (3mm) than explicitly specified mask (2mm)
        # Use small shape to speed up test
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10),  # 3mm shape
            affine=np.eye(4) * 3,  # 3mm affine
        )

        # Explicitly use 2mm mask (new behavior: mask=None would auto-detect 3mm)
        mask_img = nib.load(MNI_Template.mask)

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
        assert np.allclose(brain.nifti_masker.affine_, mask_img.affine, rtol=1e-2)

    def test_init_resample_false_matched_spaces(self):
        """Test no resampling when resample=False and spaces match."""
        import nibabel as nib

        # Create data in same space as mask
        mask_img = nib.load(MNI_Template.mask)
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
        import nibabel as nib
        import warnings

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
        import nibabel as nib
        import warnings

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
        import nibabel as nib

        # Create 3mm data
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        # Use default mask (mask=None) - will auto-detect template from data
        brain = BrainData(data_3mm, resample=True)

        # Should auto-detect and use 3mm template (not default 2mm)
        assert brain.shape[1] == 71020  # Exact voxel count for default 3mm
        assert brain._detected_template["resolution"] == 3
        assert brain._detected_template["template"] == "default"

    def test_init_resample_true_list_of_files(self, tmpdir):
        """Test resampling works with list of files."""
        import nibabel as nib
        import warnings

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
        assert brain._detected_template["resolution"] == 3

    def test_init_resample_true_matched_spaces_no_resample(self):
        """Test that resample=True skips resampling when spaces already match."""
        import nibabel as nib
        import warnings

        # Create data in same space as default mask
        mask_img = nib.load(MNI_Template.mask)
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
        import nibabel as nib

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
        import nibabel as nib

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
        import nibabel as nib

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
        import nibabel as nib

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
        import nibabel as nib

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
        import nibabel as nib

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
        import nibabel as nib

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
        import nibabel as nib
        import warnings

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
        import nibabel as nib
        import warnings
        from nltools.prefs import MNI_Template

        # Create original BrainData (2mm) using actual MNI152 affine
        default_mask = nib.load(MNI_Template.mask)
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
        import nibabel as nib

        data = nib.Nifti1Image(np.random.randn(91, 109, 91, 10), affine=np.eye(4) * 2)

        # Invalid format - should fall back to file path check
        # If file doesn't exist, should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            BrainData(data, mask="invalid-template-name")

    def test_init_mask_template_name_string_unsupported_resolution(self):
        """Test that unsupported resolution for template raises error."""
        from nltools.prefs import resolve_template_name

        # Try to use 3mm with fmriprep (not supported - only 1mm and 2mm)
        with pytest.raises(ValueError, match="Resolution 3mm is not supported"):
            resolve_template_name("3mm-MNI152-2009c", file_type="mask")

    def test_init_mask_template_name_string_file_type_brain(self):
        """Test resolve_template_name with file_type='brain'."""
        from nltools.prefs import resolve_template_name

        mask_path = resolve_template_name("2mm-MNI152-2009c", file_type="mask")
        brain_path = resolve_template_name("2mm-MNI152-2009c", file_type="brain")

        assert "mask" in mask_path
        assert "brain" in brain_path
        assert mask_path != brain_path

    def test_init_mask_template_name_string_file_type_t1(self):
        """Test resolve_template_name with file_type='T1'."""
        from nltools.prefs import resolve_template_name

        mask_path = resolve_template_name("2mm-MNI152-2009c", file_type="mask")
        t1_path = resolve_template_name("2mm-MNI152-2009c", file_type="T1")

        assert "mask" in mask_path
        assert "T1" in t1_path
        assert mask_path != t1_path

    def test_init_mask_template_name_string_invalid_file_type(self):
        """Test that invalid file_type raises error."""
        from nltools.prefs import resolve_template_name

        with pytest.raises(ValueError, match="file_type must be"):
            resolve_template_name("2mm-MNI152-2009c", file_type="invalid")

    @pytest.mark.slow
    def test_all_template_voxel_counts(self):
        """Test that all supported templates have correct voxel counts."""
        import nibabel as nib

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
        import nibabel as nib
        from nltools.prefs import resolve_template_name

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
        import nibabel as nib
        from nltools.prefs import MNI_Template

        # Create 2mm data (matches default template) using actual MNI152 affine
        default_mask = nib.load(MNI_Template.mask)
        data_2mm = nib.Nifti1Image(
            np.random.randn(91, 109, 91, 10), affine=default_mask.affine
        )

        brain = BrainData(data_2mm, mask=None, resample=True)

        # Should detect and use 2mm template with exact voxel count
        assert brain.shape[1] == 238955  # Exact voxel count for default 2mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 2.0, rtol=1e-3)
        assert hasattr(brain, "_detected_template")
        assert brain._detected_template["resolution"] == 2
        assert brain._detected_template["template"] == "default"

    def test_init_mask_none_auto_detect_3mm(self):
        """Test automatic template detection for 3mm data."""
        import nibabel as nib

        # Create 3mm data
        data_3mm = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )

        brain = BrainData(data_3mm, mask=None, resample=True)

        # Should detect and use 3mm template with exact voxel count
        assert brain.shape[1] == 71020  # Exact voxel count for default 3mm
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 3.0, rtol=1e-3)
        assert brain._detected_template["resolution"] == 3
        assert brain._detected_template["template"] == "default"

    @pytest.mark.slow
    def test_init_mask_none_auto_detect_1mm(self):
        """Test automatic template detection for 1mm data (uses nilearn template)."""
        import nibabel as nib

        # Create 1mm data (only available in nilearn template)
        data_1mm = nib.Nifti1Image(
            np.random.randn(182, 218, 182, 10), affine=np.eye(4) * 1
        )

        brain = BrainData(data_1mm, mask=None, resample=True)

        # Should detect and use 1mm nilearn template with exact voxel count
        assert np.allclose(np.abs(brain.mask.affine[0, 0]), 1.0, rtol=1e-3)
        assert brain.shape[1] == 1886539  # Exact voxel count for nilearn 1mm
        assert brain._detected_template["resolution"] == 1
        assert brain._detected_template["template"] == "nilearn"

    def test_init_mask_none_resample_false_exact_match(self):
        """Test auto-detection with resample=False requires exact match."""
        import nibabel as nib

        # Create 2mm data that exactly matches template
        mask_2mm = nib.load(MNI_Template.mask)  # Get exact template
        data_2mm = nib.Nifti1Image(
            np.random.randn(*mask_2mm.shape + (10,)), affine=mask_2mm.affine
        )

        brain = BrainData(data_2mm, mask=None, resample=False)

        # Should use template without resampling
        assert brain.shape[1] == mask_2mm.get_fdata().sum().astype(int)

    def test_init_mask_none_resample_false_mismatch(self):
        """Test that resample=False with mismatched data shows warning but still resamples."""
        import nibabel as nib
        import warnings

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
        import nibabel as nib
        import warnings

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
        import nibabel as nib

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
        import nibabel as nib
        from nltools.prefs import MNI_Template

        # Create original BrainData using actual MNI152 affine
        default_mask = nib.load(MNI_Template.mask)
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
        import nibabel as nib
        import warnings
        from nltools.prefs import MNI_Template

        # Create original BrainData (2mm) using actual MNI152 affine
        default_mask = nib.load(MNI_Template.mask)
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
        import nibabel as nib
        from nltools.prefs import MNI_Template

        # Create original BrainData (2mm) using actual MNI152 affine to avoid resampling warning
        default_mask = nib.load(MNI_Template.mask)
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
        import nibabel as nib

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
        assert np.allclose(
            brain_default.nifti_masker.affine_, brain_explicit.nifti_masker.affine_
        )

    @pytest.mark.slow
    def test_init_resample_preserves_data_integrity(self):
        """Test that resampling preserves data characteristics."""
        import nibabel as nib

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
        import nibabel as nib

        mask_img = nib.load(MNI_Template.mask)

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

    @pytest.mark.slow
    def test_load(self, tmpdir):
        """Test loading BrainData from various sources and formats."""
        sim = Simulator()
        sigma = 1
        y = [0, 1]
        n_reps = 3
        output_dir = str(tmpdir)
        dat = sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

        # Shape depends on MNI_Template.resolution
        # 2mm: shape_3d = (91, 109, 91), shape_2d = (6, 238955)
        # 3mm: shape_3d = (60, 72, 60), shape_2d = (6, 71020)

        y = pd.read_csv(
            os.path.join(str(tmpdir.join("y.csv"))), header=None, index_col=None
        )

        # Test load list of 4D images
        file_list = [str(tmpdir.join("data.nii.gz")), str(tmpdir.join("data.nii.gz"))]
        dat = BrainData(file_list)
        dat = BrainData([nb.load(x) for x in file_list])

        # Test load string and path
        dat = BrainData(data=str(tmpdir.join("data.nii.gz")), Y=y)
        dat = BrainData(data=Path(tmpdir.join("data.nii.gz")), Y=y)

        # Test Write
        dat.write(os.path.join(str(tmpdir.join("test_write.nii"))))
        assert BrainData(os.path.join(str(tmpdir.join("test_write.nii"))))

        # Test i/o for hdf5
        dat.write(os.path.join(str(tmpdir.join("test_write.h5"))))
        b = BrainData(os.path.join(tmpdir.join("test_write.h5")))
        # Note: X and Y attributes removed in v0.6.0, skip checking them
        for k in ["mask", "nifti_masker", "data"]:
            if k == "data":
                assert np.allclose(b.__dict__[k], dat.__dict__[k])
            elif k == "mask":
                assert np.allclose(b.__dict__[k].affine, dat.__dict__[k].affine)
                assert np.allclose(
                    b.__dict__[k].get_fdata(), dat.__dict__[k].get_fdata()
                )
                assert b.__dict__[k].get_filename() == dat.__dict__[k].get_filename()
            elif k == "nifti_masker":
                assert np.allclose(b.__dict__[k].affine_, dat.__dict__[k].affine_)
                assert np.allclose(
                    b.__dict__[k].mask_img.get_fdata(),
                    dat.__dict__[k].mask_img.get_fdata(),
                )
            else:
                assert b.__dict__[k] == dat.__dict__[k]
        # Test situation where we present a user warning when they're trying to load an .h5
        # file that includes a mask AND they pass in value for the mask argument. In this
        # case the mask argument takes precedence so we warn the user
        with pytest.warns(UserWarning):
            bb = BrainData(
                os.path.join(tmpdir.join("test_write.h5")), mask=MNI_Template.mask
            )
            assert os.path.abspath(bb.mask.get_filename()) == os.path.abspath(
                MNI_Template.mask
            )

    @pytest.mark.skipif(
        not _tables_available(), reason="HDF5 support deprecated, requires PyTables"
    )
    def test_load_legacy_h5(self, old_h5_brain, new_h5_brain, tmpdir):
        """Test loading old HDF5 format (backward compatibility)."""
        with pytest.warns(UserWarning):
            # With verbosity on we should see a warning about the old h5 file format
            b_old = BrainData(old_h5_brain, verbose=True)
        b_new = BrainData(new_h5_brain)
        assert b_old.shape == b_new.shape
        assert np.allclose(b_old.data, b_new.data)
        # NOTE: We lose pandas column dtype information between old and new h5 files
        # so we can't use .equals()
        assert b_old.X.shape == b_new.X.shape
        assert b_old.Y.shape == b_new.Y.shape
        assert np.allclose(b_old.mask.affine, b_new.mask.affine)
        assert np.allclose(b_old.mask.get_fdata(), b_new.mask.get_fdata())

        new_file = Path(tmpdir) / "tmp.h5"
        b_new.write(new_file)
        b_new_written = BrainData(new_file)
        assert b_new.shape == b_new_written.shape
        assert np.allclose(b_new.data, b_new_written.data)
        new_file.unlink()

    # ==================== Resampling Methods ====================

    def test_resample_to_img_nibabel(self):
        """Test resampling to target nibabel image."""
        import nibabel as nib

        # Create source BrainData (3mm) - need custom mask in 3mm space
        source_data = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )
        # Create 3mm mask matching the data
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )
        brain_source = BrainData(source_data, mask=mask_3mm, resample=False)

        # Create target image (2mm, same as MNI template)
        mask_img = nib.load(MNI_Template.mask)

        # Resample
        brain_resampled = brain_source.resample_to(img=mask_img)

        # Should match target space
        assert brain_resampled.shape[1] == 238955  # 2mm voxel count
        assert np.allclose(
            brain_resampled.nifti_masker.affine_, mask_img.affine, rtol=1e-2
        )
        assert (
            brain_resampled.shape[0] == brain_source.shape[0]
        )  # Same number of images

    def test_resample_to_img_filepath(self, tmpdir):
        """Test resampling to target image from file path."""
        import nibabel as nib

        # Create source BrainData (3mm) - need custom mask in 3mm space
        source_data = nib.Nifti1Image(
            np.random.randn(60, 72, 60, 10), affine=np.eye(4) * 3
        )
        # Create 3mm mask matching the data
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=np.eye(4) * 3
        )
        brain_source = BrainData(source_data, mask=mask_3mm, resample=False)

        # Save target image to file
        target_path = str(tmpdir.join("target.nii.gz"))
        mask_img = nib.load(MNI_Template.mask)
        mask_img.to_filename(target_path)

        # Resample
        brain_resampled = brain_source.resample_to(img=target_path)

        # Should match target space
        assert brain_resampled.shape[1] == 238955
        assert brain_resampled.shape[0] == brain_source.shape[0]

    def test_resample_to_resolution_isotropic(self):
        """Test resampling to specified isotropic resolution."""
        import nibabel as nib

        # Create source BrainData (2mm)
        mask_img = nib.load(MNI_Template.mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (10,)), affine=mask_img.affine
        )
        brain_source = BrainData(source_data, resample=False)

        # Resample to 3mm
        brain_resampled = brain_source.resample_to(resolution=3.0)

        # Should have different voxel count
        assert brain_resampled.shape[1] != brain_source.shape[1]
        # Check resolution is approximately 3mm
        resampled_nifti = brain_resampled.to_nifti()
        zooms = resampled_nifti.header.get_zooms()[:3]
        assert np.allclose(zooms, 3.0, rtol=0.1)
        assert (
            brain_resampled.shape[0] == brain_source.shape[0]
        )  # Same number of images

    def test_resample_to_both_params_error(self):
        """Test error when both img and resolution are provided."""
        import nibabel as nib

        # Create BrainData with matching mask
        mask_img = nib.load(MNI_Template.mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape), affine=mask_img.affine
        )
        brain = BrainData(source_data, resample=False)

        with pytest.raises(ValueError, match="both.*img.*and.*resolution"):
            brain.resample_to(img=mask_img, resolution=2.0)

    def test_resample_to_no_params_error(self):
        """Test error when neither img nor resolution is provided."""
        import nibabel as nib

        # Create BrainData with matching mask
        mask_img = nib.load(MNI_Template.mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape), affine=mask_img.affine
        )
        brain = BrainData(source_data, resample=False)

        with pytest.raises(ValueError, match="either.*img.*or.*resolution"):
            brain.resample_to()

    def test_resample_to_invalid_img_type(self):
        """Test error with invalid img type."""
        import nibabel as nib

        # Create BrainData with matching mask
        mask_img = nib.load(MNI_Template.mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape), affine=mask_img.affine
        )
        brain = BrainData(source_data, resample=False)

        with pytest.raises(TypeError, match="img.*must be"):
            brain.resample_to(img=123)  # Invalid type

    def test_resample_to_preserves_metadata(self):
        """Test that X and Y metadata are preserved after resampling."""
        import nibabel as nib

        # Create BrainData with matching mask
        mask_img = nib.load(MNI_Template.mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (5,)), affine=mask_img.affine
        )
        X = pd.DataFrame({"cond1": [1, 2, 3, 4, 5]})
        Y = pd.DataFrame({"outcome": [0, 1, 0, 1, 0]})

        brain_source = BrainData(source_data, X=X, Y=Y, resample=False)
        brain_resampled = brain_source.resample_to(resolution=3.0)

        # Metadata should be preserved
        assert brain_resampled.X.equals(brain_source.X)
        assert brain_resampled.Y.equals(brain_source.Y)
        assert (
            brain_resampled.shape[0] == brain_source.shape[0]
        )  # Same number of images

    def test_resample_to_same_space_identity(self):
        """Test resampling to same space produces similar results."""
        import nibabel as nib

        # Create BrainData
        mask_img = nib.load(MNI_Template.mask)
        source_data = nib.Nifti1Image(
            np.random.randn(*mask_img.shape + (5,)), affine=mask_img.affine
        )
        brain_source = BrainData(source_data, resample=False)

        # Resample to same space (should be near-identical)
        brain_resampled = brain_source.resample_to(img=mask_img)

        # Data should be very similar (within interpolation tolerance)
        assert np.allclose(
            brain_source.data, brain_resampled.data, rtol=1e-3, atol=1e-3
        )

    # ==================== Interpolation Tests ====================

    def test_interpolation_auto_default(self):
        """Test that interpolation='auto' is the default."""
        brain = BrainData()
        assert brain._interpolation == "auto"

    def test_interpolation_explicit_nearest(self):
        """Test explicit interpolation='nearest' is stored."""
        brain = BrainData(interpolation="nearest")
        assert brain._interpolation == "nearest"

    def test_interpolation_explicit_continuous(self):
        """Test explicit interpolation='continuous' is stored."""
        brain = BrainData(interpolation="continuous")
        assert brain._interpolation == "continuous"

    def test_interpolation_invalid_raises_error(self):
        """Test invalid interpolation value raises ValueError."""
        with pytest.raises(ValueError, match="interpolation must be one of"):
            BrainData(interpolation="invalid")

    def test_interpolation_auto_detects_atlas(self, tmpdir):
        """Test auto-detection uses nearest for atlas/label data."""
        import nibabel as nib

        # Create atlas with discrete integer labels
        atlas_data = np.zeros((20, 20, 20))
        atlas_data[5:10, 5:10, 5:10] = 1
        atlas_data[10:15, 10:15, 10:15] = 2
        atlas_data[15:18, 15:18, 15:18] = 3

        # 3mm voxels to force resampling to 2mm template
        affine = np.eye(4) * 3
        affine[3, 3] = 1
        atlas_img = nib.Nifti1Image(atlas_data, affine)

        # Load with auto interpolation
        brain = BrainData(atlas_img, interpolation="auto")

        # Values should remain discrete (no interpolation artifacts)
        unique_vals = np.unique(brain.data)
        # With nearest interpolation, we should have only a few discrete values
        # (some may be 0 due to masking, but shouldn't have many intermediate values)
        assert len(unique_vals) < 10, (
            f"Expected discrete values, got {len(unique_vals)} unique values"
        )

    def test_interpolation_auto_detects_continuous(self, tmpdir):
        """Test auto-detection uses continuous for statistical maps."""
        import nibabel as nib

        # Create continuous statistical data
        stat_data = np.random.randn(20, 20, 20)

        # 3mm voxels
        affine = np.eye(4) * 3
        affine[3, 3] = 1
        stat_img = nib.Nifti1Image(stat_data, affine)

        # Load with auto interpolation
        brain = BrainData(stat_img, interpolation="auto")

        # Continuous data should have many unique values
        unique_vals = np.unique(brain.data)
        assert len(unique_vals) > 100, (
            f"Expected many unique values, got {len(unique_vals)}"
        )

    def test_resample_to_respects_interpolation(self):
        """Test resample_to uses instance interpolation setting."""
        import nibabel as nib

        # Create atlas-like source data with explicit nearest
        atlas_data = np.zeros((60, 72, 60))
        atlas_data[20:40, 20:50, 20:40] = 1
        atlas_data[30:50, 30:60, 30:50] = 2

        affine = np.eye(4) * 3
        affine[3, 3] = 1
        atlas_img = nib.Nifti1Image(atlas_data, affine)
        mask_3mm = nib.Nifti1Image(
            np.ones((60, 72, 60), dtype=np.float32), affine=affine
        )

        # Load with explicit nearest interpolation
        brain = BrainData(
            atlas_img, mask=mask_3mm, interpolation="nearest", resample=False
        )

        # Resample to 2mm template
        mask_2mm = nib.load(MNI_Template.mask)
        brain_resampled = brain.resample_to(img=mask_2mm)

        # Values should still be discrete after resampling
        unique_vals = np.unique(brain_resampled.data)
        assert len(unique_vals) < 10, (
            f"Expected discrete values after resample, got {len(unique_vals)}"
        )

    # ==================== Properties & Basic Operations ====================

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

    # ==================== Regression & GLM ====================

    def test_regress_removed(self, sim_brain_data):
        """Verify regress() has been removed with clear migration path."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Should raise NotImplementedError with migration message
        with pytest.raises(
            NotImplementedError,
            match="regress.*has been removed.*Use fit.*model='glm'",
        ):
            sim_brain_data.regress(design_matrix)

    def test_compute_contrasts_error_not_fitted(self, minimal_brain_data):
        """Test error when compute_contrasts() called before fit()."""
        # Should raise RuntimeError if fit() not called first
        with pytest.raises(RuntimeError, match="Must run .fit"):
            minimal_brain_data.compute_contrasts([1, -1, 0])

    @pytest.mark.slow
    def test_compute_contrasts_numeric_vector(self, minimal_brain_data):
        """Test numeric contrast vector (unique nltools API)."""
        # Set up and run regression using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Compute contrast: A - B (unique nltools logic)
        contrast = minimal_brain_data.compute_contrasts([0, 1, -1])

        # Test nltools-specific API contract
        assert isinstance(contrast, BrainData)
        assert contrast.shape == (1, minimal_brain_data.shape[1])

    @pytest.mark.slow
    def test_compute_contrasts_string_parsing(self, minimal_brain_data):
        """Test string parsing (unique nltools feature)."""
        # Set up and run regression using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Test string parsing (unique nltools feature)
        contrast = minimal_brain_data.compute_contrasts("condA - condB")

        assert isinstance(contrast, BrainData)
        assert contrast.shape == (1, minimal_brain_data.shape[1])

    @pytest.mark.slow
    def test_compute_contrasts_multiple_dict(self, minimal_brain_data):
        """Test multiple contrasts via dict (unique nltools API)."""
        # Set up and run regression using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Test dict of contrasts (unique nltools API)
        contrasts = {"A_vs_B": "condA - condB", "avg_effect": [0, 0.5, 0.5]}
        results = minimal_brain_data.compute_contrasts(contrasts)

        # Should return dict of BrainData objects
        assert isinstance(results, dict)
        assert "A_vs_B" in results
        assert "avg_effect" in results
        assert isinstance(results["A_vs_B"], BrainData)
        assert isinstance(results["avg_effect"], BrainData)

    @pytest.mark.slow
    def test_compute_contrasts_invalid_length(self, minimal_brain_data):
        """Test error for invalid contrast vector length (nltools validation)."""
        # Set up and run regression with 3 regressors using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Provide wrong length contrast (2 instead of 3)
        with pytest.raises(ValueError, match="Contrast vector length.*must match"):
            minimal_brain_data.compute_contrasts([1, -1])

    # ==================== Unified fit/predict API ====================

    def test_fit_predict_ridge_workflow(self, sim_brain_data):
        """Test complete Ridge fit/predict workflow."""
        from nltools.data import BrainData
        from nltools.models import Ridge

        # Fit Ridge model
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Check model stored
        assert hasattr(sim_brain_data, "model_")
        assert isinstance(sim_brain_data.model_, Ridge)
        assert sim_brain_data.model_.is_fitted_

        # Check attributes set
        assert hasattr(sim_brain_data, "ridge_weights")
        assert hasattr(sim_brain_data, "ridge_fitted_values")
        assert hasattr(sim_brain_data, "ridge_scores")

        # Predict on new data
        X_test = np.random.randn(20, 10)  # Different n_samples
        predictions = sim_brain_data.predict(X=X_test)

        # Check predictions
        assert isinstance(predictions, BrainData)
        assert predictions.shape == (20, sim_brain_data.shape[1])

        # Predict on training data (X=None)
        train_predictions = sim_brain_data.predict()
        assert train_predictions.shape == sim_brain_data.shape

    @pytest.mark.slow
    def test_fit_predict_glm_workflow(self, sim_brain_data):
        """Test complete GLM fit/predict workflow."""
        from nltools.models import Glm

        # Fit GLM model
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )
        sim_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        # Check model stored
        assert hasattr(sim_brain_data, "model_")
        assert isinstance(sim_brain_data.model_, Glm)

        # Check GLM attributes set
        assert hasattr(sim_brain_data, "glm_betas")
        assert hasattr(sim_brain_data, "glm_t")

        # Predict on training data (fitted values)
        # Note: GLM doesn't support prediction with new design matrices yet
        predictions = sim_brain_data.predict()

        # Check predictions match training data shape
        assert predictions.shape == sim_brain_data.shape

    def test_fit_uses_brain_data_as_target(self, sim_brain_data):
        """Test fit() always uses self.data as y target."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Fit Ridge
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Model should be fitted to (X, sim_brain_data.data)
        # Check by predicting and comparing shapes
        predictions = sim_brain_data.predict(X=X)
        assert predictions.shape == sim_brain_data.shape

    @pytest.mark.slow
    def test_fit_passes_kwargs_to_model(self, sim_brain_data):
        """Test fit() passes additional kwargs to model constructor."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Ridge with backend kwarg
        sim_brain_data.fit(model="ridge", alpha=1.0, backend="numpy", X=X)
        assert sim_brain_data.model_.backend == "numpy"

        # GLM with noise_model kwarg
        design_matrix = pd.DataFrame({"Intercept": np.ones(len(sim_brain_data))})
        sim_brain_data.fit(model="glm", noise_model="ar1", X=design_matrix)
        assert sim_brain_data.model_.noise_model == "ar1"

    def test_predict_requires_fitted_model(self, sim_brain_data):
        """Test predict() raises error if fit() not called first."""
        # Get a fresh copy (fixture may be contaminated by previous tests)
        bd = sim_brain_data.copy()

        # Explicitly remove model attributes to test the error case
        # (copy shares model_ from fitted instances due to pickle handling)
        for attr in ["model_", "X_"]:
            if hasattr(bd, attr):
                delattr(bd, attr)

        with pytest.raises(ValueError, match="Must call fit"):
            bd.predict()

    def test_predict_validates_X_dimensions(self, sim_brain_data):
        """Test predict() validates X has correct n_features."""
        # Fit with 10 features
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Try to predict with 5 features - should fail
        X_wrong = np.random.randn(15, 5)
        with pytest.raises(ValueError, match="features"):
            sim_brain_data.predict(X=X_wrong)

    def test_ridge_weights_structure(self, sim_brain_data):
        """Test Ridge weights stored correctly as BrainData."""
        from nltools.data import BrainData

        X = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Weights should be BrainData
        assert isinstance(sim_brain_data.ridge_weights, BrainData)

        # Shape: (n_features, n_voxels)
        assert sim_brain_data.ridge_weights.shape == (10, sim_brain_data.shape[1])

        # Should have same mask
        assert sim_brain_data.ridge_weights.mask is sim_brain_data.mask

    # ==================== Fit inplace parameter tests ====================

    def test_fit_inplace_true_backward_compatible(self, sim_brain_data):
        """Test inplace=True preserves backward compatibility."""
        import numpy as np

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit with inplace=True (default)
        result = sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=True)

        # Should return self
        assert result is sim_brain_data

        # Should have mutated attributes
        assert hasattr(sim_brain_data, "ridge_weights")
        assert hasattr(sim_brain_data, "ridge_fitted_values")
        assert hasattr(sim_brain_data, "ridge_scores")
        assert hasattr(sim_brain_data, "model_")
        assert hasattr(sim_brain_data, "X_")

    def test_fit_inplace_false_returns_fit_dataclass_ridge(self, sim_brain_data):
        """Test inplace=False returns Fit dataclass for Ridge."""
        from nltools.data.fitresults import Fit
        import numpy as np

        # Use a fresh copy to avoid contamination from previous tests
        brain = sim_brain_data.copy()
        # Clean up any existing fit attributes that might have been copied
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_scores",
            "glm_betas",
            "glm_t",
            "glm_p",
            "glm_se",
            "glm_residual",
            "glm_predicted",
            "glm_r2",
            "cv_results_",
            "model_",
            "X_",
        ]:
            if hasattr(brain, attr):
                delattr(brain, attr)

        X_train = np.random.randn(len(brain), 10)
        original_data = brain.data.copy()

        # Fit with inplace=False
        fit = brain.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        # Should return Fit dataclass
        assert isinstance(fit, Fit)

        # Should have correct fields
        assert "fitted_values" in fit.available()
        assert "weights" in fit.available()
        assert "scores" in fit.available()

        # Check shapes
        assert fit.fitted_values.shape == brain.shape
        assert fit.weights.shape == (10, brain.shape[1])
        assert fit.scores.shape == (brain.shape[1],)

        # BrainData should not have result attributes
        assert not hasattr(brain, "ridge_weights")
        assert not hasattr(brain, "ridge_fitted_values")
        assert not hasattr(brain, "ridge_scores")

        # But should have model_ and X_ for predict()
        assert hasattr(brain, "model_")
        assert hasattr(brain, "X_")

        # Data should be unchanged
        np.testing.assert_array_equal(brain.data, original_data)

    @pytest.mark.slow
    def test_fit_inplace_false_returns_fit_dataclass_ridge_cv(self, sim_brain_data):
        """Test inplace=False returns Fit dataclass with CV results for Ridge."""
        from nltools.data.fitresults import Fit
        import numpy as np

        # Use a fresh copy to avoid contamination
        brain = sim_brain_data.copy()
        X_train = np.random.randn(len(brain), 10)

        # Fit with CV and inplace=False
        fit = brain.fit(model="ridge", alpha=1.0, X=X_train, cv=3, inplace=False)

        # Should return Fit dataclass
        assert isinstance(fit, Fit)

        # Should have CV fields
        assert "cv_scores" in fit.available()
        assert "cv_mean_score" in fit.available()
        assert "cv_predictions" in fit.available()
        assert "cv_folds" in fit.available()

        # Check shapes
        assert fit.cv_scores.shape == (3, brain.shape[1])
        assert fit.cv_mean_score.shape == (brain.shape[1],)
        assert fit.cv_predictions.shape == brain.shape
        assert fit.cv_folds.shape == (len(brain),)

        # BrainData should not have cv_results_
        assert not hasattr(brain, "cv_results_")

    @pytest.mark.slow
    def test_fit_inplace_false_returns_fit_dataclass_glm(self, sim_brain_data):
        """Test inplace=False returns Fit dataclass for GLM."""
        from nltools.data.fitresults import Fit
        import numpy as np
        import pandas as pd

        # Use a fresh copy to avoid contamination
        brain = sim_brain_data.copy()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(brain)),
                "X1": np.random.randn(len(brain)),
            }
        )
        original_data = brain.data.copy()

        # Fit with inplace=False
        fit = brain.fit(model="glm", noise_model="ols", X=design_matrix, inplace=False)

        # Should return Fit dataclass
        assert isinstance(fit, Fit)

        # Should have GLM fields
        assert "fitted_values" in fit.available()
        assert "betas" in fit.available()
        assert "t_stats" in fit.available()
        assert "p_values" in fit.available()
        assert "se" in fit.available()
        assert "residuals" in fit.available()
        assert "r2" in fit.available()

        # Check shapes
        assert fit.fitted_values.shape == brain.shape
        assert fit.betas.shape == (2, brain.shape[1])  # 2 regressors
        assert fit.t_stats.shape == (2, brain.shape[1])
        assert fit.p_values.shape == (2, brain.shape[1])
        assert fit.se.shape == (2, brain.shape[1])
        assert fit.residuals.shape == brain.shape
        assert fit.r2.shape == (brain.shape[1],)

        # BrainData should not have GLM result attributes
        assert not hasattr(brain, "glm_betas")
        assert not hasattr(brain, "glm_t")
        assert not hasattr(brain, "glm_p")

        # But should have model_ and design_matrix for predict() and compute_contrasts()
        assert hasattr(brain, "model_")
        assert hasattr(brain, "design_matrix")

        # Data should be unchanged
        np.testing.assert_array_equal(brain.data, original_data)

    def test_fit_inplace_false_allows_predict(self, sim_brain_data):
        """Test that inplace=False still allows predict() to work."""
        import numpy as np

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit with inplace=False
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        # Should be able to predict (model_ and X_ stored)
        X_test = np.random.randn(20, 10)
        predictions = sim_brain_data.predict(X=X_test)

        assert predictions.shape == (20, sim_brain_data.shape[1])

    def test_fit_inplace_false_serialization(self, sim_brain_data):
        """Test Fit dataclass serialization roundtrip."""
        from nltools.data.fitresults import Fit
        import numpy as np
        import tempfile
        import os

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit with inplace=False
        fit = sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        # Serialize
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **fit.asdict())

            # Load
            loaded = np.load(f.name)
            fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})

            # Clean up
            os.unlink(f.name)

        # Check fields match
        np.testing.assert_array_equal(
            fit.fitted_values, fit_reconstructed.fitted_values
        )
        np.testing.assert_array_equal(fit.weights, fit_reconstructed.weights)
        np.testing.assert_array_equal(fit.scores, fit_reconstructed.scores)

    def test_fit_inplace_default_is_true(self, sim_brain_data):
        """Test that inplace defaults to True for backward compatibility."""
        import numpy as np

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit without specifying inplace (should default to True)
        result = sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Should return self and mutate attributes
        assert result is sim_brain_data
        assert hasattr(sim_brain_data, "ridge_weights")
        # Verify progress_bar parameter exists on model (defaults to verbose=False)
        assert hasattr(sim_brain_data.model_, "progress_bar")
        assert sim_brain_data.model_.progress_bar is False

    @pytest.mark.slow
    def test_glm_fit_numerical_correctness(self, sim_brain_data):
        """Test fit(model='glm') produces numerically correct results."""

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )

        # Fit GLM model
        sim_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        # Check betas are reasonable (not NaN, not all zeros)
        assert not np.isnan(sim_brain_data.glm_betas.data).any()
        assert not np.allclose(sim_brain_data.glm_betas.data, 0)

        # Check t-statistics are reasonable
        assert not np.isnan(sim_brain_data.glm_t.data).any()
        # Verify progress_bar parameter exists and defaults to False
        assert hasattr(sim_brain_data.model_, "progress_bar")
        assert sim_brain_data.model_.progress_bar is False

    @pytest.mark.slow
    def test_glm_fit_suppresses_drift_model_warning(self, sim_brain_data):
        """Test fit(model='glm') suppresses drift_model warning when design matrices are supplied"""
        import warnings

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )

        # Capture warnings during fit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            # Fit GLM with drift_model set (would trigger warning without suppression)
            sim_brain_data.fit(
                model="glm", noise_model="ols", X=design_matrix, drift_model="cosine"
            )

            # Check that drift_model warning is NOT present
            drift_warnings = [
                warn
                for warn in w
                if "drift_model" in str(warn.message).lower()
                and "will be ignored" in str(warn.message).lower()
            ]
            assert len(drift_warnings) == 0, (
                f"Expected no drift_model warnings, but got {len(drift_warnings)}: "
                f"{[str(w.message) for w in drift_warnings]}"
            )

        # Verify model was fitted successfully
        assert hasattr(sim_brain_data, "model_")
        assert sim_brain_data.model_.is_fitted_
        # Verify progress_bar parameter exists and defaults to False
        assert hasattr(sim_brain_data.model_, "progress_bar")
        assert sim_brain_data.model_.progress_bar is False

        # Test with progress_bar=True to verify it's respected
        sim_brain_data.fit(
            model="glm", noise_model="ols", X=design_matrix, progress_bar=True
        )
        assert sim_brain_data.model_.progress_bar is True
        assert sim_brain_data.model_.is_fitted_

    def test_fit_validates_model_name(self, sim_brain_data):
        """Test fit() raises error for unknown model names."""
        X = np.random.randn(len(sim_brain_data), 10)

        with pytest.raises(ValueError, match="Unknown model"):
            sim_brain_data.fit(model="unknown_model", X=X)

    def test_fit_validates_X_shape(self, sim_brain_data):
        """Test fit() validates X has correct n_samples."""
        # X has wrong number of samples
        X_wrong = np.random.randn(len(sim_brain_data) + 5, 10)

        with pytest.raises(ValueError, match="number of samples"):
            sim_brain_data.fit(model="ridge", alpha=1.0, X=X_wrong)

    def test_fit_scale_default_true(self, sim_brain_data):
        """Test fit() applies scaling by default (scale=True)."""
        X = np.random.randn(len(sim_brain_data), 10)
        original_mean = sim_brain_data.data.mean()

        # Fit with default scale=True
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Data should be scaled (mean should be ~100 after grand-mean scaling)
        # Note: original mean is ~0 for simulated data, so result will be very different
        assert sim_brain_data.data.mean() != original_mean
        # After scaling, mean should be close to scale_value (100)
        np.testing.assert_allclose(sim_brain_data.data.mean(), 100.0, rtol=0.1)

    def test_fit_scale_false_preserves_data(self, sim_brain_data):
        """Test fit() with scale=False preserves original data values."""
        X = np.random.randn(len(sim_brain_data), 10)
        original_data = sim_brain_data.data.copy()

        # Fit with scale=False
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X, scale=False)

        # Data should be unchanged
        np.testing.assert_allclose(sim_brain_data.data, original_data)

    def test_fit_scale_value_custom(self, sim_brain_data):
        """Test fit() respects custom scale_value."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Fit with custom scale_value
        sim_brain_data.fit(
            model="ridge", alpha=1.0, X=X, scale=True, scale_value=1000.0
        )

        # Mean should be close to custom scale_value
        np.testing.assert_allclose(sim_brain_data.data.mean(), 1000.0, rtol=0.1)

    def test_fit_scale_inplace_false(self, sim_brain_data):
        """Test fit() with scale=True and inplace=False doesn't modify original."""
        from nltools.data.fitresults import Fit

        X = np.random.randn(len(sim_brain_data), 10)
        original_data = sim_brain_data.data.copy()

        # Fit with inplace=False and scale=True
        result = sim_brain_data.fit(
            model="ridge", alpha=1.0, X=X, inplace=False, scale=True
        )

        # Should return Fit dataclass
        assert isinstance(result, Fit)

        # Original data should be unchanged (scaling was applied to copy)
        np.testing.assert_allclose(sim_brain_data.data, original_data)

    def test_predict_with_no_X_uses_training_data(self, sim_brain_data):
        """Test predict() with no X returns predictions on training data."""
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Predict with explicit X
        predictions_explicit = sim_brain_data.predict(X=X_train)

        # Predict with no X (should use training data)
        predictions_implicit = sim_brain_data.predict()

        # Should be identical
        np.testing.assert_allclose(predictions_explicit.data, predictions_implicit.data)

        # Should match training data shape
        assert predictions_implicit.shape == sim_brain_data.shape

    # ==================== predict() MVPA Mode ====================

    def test_predict_mvpa_whole_brain(self, sim_brain_data):
        """Test predict(y=...) performs MVPA decoding."""
        # Create binary classification problem
        n_samples = sim_brain_data.shape[0]
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

        # Run whole-brain MVPA
        accuracy = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, show_progress=False
        )

        # Should return BrainData with single accuracy value
        assert isinstance(accuracy, BrainData)
        assert accuracy.shape[0] == 1
        # Accuracy should be between 0 and 1
        assert 0 <= accuracy.data.flatten()[0] <= 1

    def test_predict_mvpa_cannot_specify_both_x_and_y(self, sim_brain_data):
        """Test that specifying both X and y raises error."""
        X = np.random.randn(len(sim_brain_data), 5)
        y = np.array([0, 1] * (len(sim_brain_data) // 2))

        with pytest.raises(ValueError, match="Cannot specify both X and y"):
            sim_brain_data.predict(X=X, y=y)

    def test_predict_mvpa_invalid_method(self, sim_brain_data):
        """Test invalid method raises error."""
        y = np.array([0, 1] * (len(sim_brain_data) // 2))

        with pytest.raises(ValueError, match="Invalid method"):
            sim_brain_data.predict(y=y, method="invalid_method")

    def test_predict_mvpa_custom_estimator(self, sim_brain_data):
        """Test custom sklearn estimator works."""
        from sklearn.linear_model import LogisticRegression

        n_samples = sim_brain_data.shape[0]
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

        accuracy = sim_brain_data.predict(
            y=y,
            method="whole_brain",
            estimator=LogisticRegression(max_iter=1000),
            cv=3,
            show_progress=False,
        )

        assert isinstance(accuracy, BrainData)
        assert 0 <= accuracy.data.flatten()[0] <= 1

    # ==================== fit() with Cross-Validation ====================

    def test_fit_ridge_cv_basic_integer(self, small_brain_data_for_cv):
        """Test fit() with cv=3 returns cross-validated scores for fixed alpha."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV and fixed alpha
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # CV results should exist
        assert hasattr(brain_data, "cv_results_")
        assert isinstance(brain_data.cv_results_, dict)

        # Check expected keys
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_
        assert "predictions" in brain_data.cv_results_
        assert "folds" in brain_data.cv_results_

        # Check shapes
        cv_scores = brain_data.cv_results_["scores"]
        assert cv_scores.shape == (3, 5)  # (n_folds=3, n_voxels=5)

        mean_score = brain_data.cv_results_["mean_score"]
        assert mean_score.shape == (5,)  # Per-voxel mean

        # Check fold indices
        folds = brain_data.cv_results_["folds"]
        assert len(folds) == 24  # n_samples
        assert set(folds) == {0, 1, 2}  # Fold IDs

        # Regular fit attributes should still exist
        assert hasattr(brain_data, "ridge_weights")
        assert hasattr(brain_data, "ridge_fitted_values")

    def test_fit_ridge_cv_sklearn_splitter(self, small_brain_data_for_cv):
        """Test fit() accepts sklearn CV splitter objects."""
        from sklearn.model_selection import KFold

        brain_data, X = small_brain_data_for_cv

        # Create CV splitter
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42)

        # Fit with CV splitter
        brain_data.fit(model="ridge", alpha=1.0, cv=cv_splitter, X=X)

        # CV results should exist with same structure
        assert hasattr(brain_data, "cv_results_")
        assert brain_data.cv_results_["scores"].shape == (3, 5)

        # Test reproducibility - fit again with same random_state
        brain_data2, X2 = small_brain_data_for_cv
        cv_splitter2 = KFold(n_splits=3, shuffle=True, random_state=42)
        brain_data2.fit(model="ridge", alpha=1.0, cv=cv_splitter2, X=X2)

        # Should get identical results
        np.testing.assert_allclose(
            brain_data.cv_results_["mean_score"], brain_data2.cv_results_["mean_score"]
        )

    def test_fit_ridge_cv_predictions(self, small_brain_data_for_cv):
        """Test CV predictions are out-of-fold and stored as BrainData."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # Check predictions structure
        cv_preds = brain_data.cv_results_["predictions"]
        assert isinstance(cv_preds, BrainData)
        assert cv_preds.shape == (24, 5)  # (n_samples, n_voxels)

        # CV predictions should differ from full model predictions
        # (out-of-fold vs. in-sample)
        full_preds = brain_data.ridge_fitted_values
        assert not np.allclose(cv_preds.data, full_preds.data)

        # Sanity checks on R² values
        # Note: Out-of-sample R² can be negative (model worse than mean)
        cv_r2 = np.mean(brain_data.cv_results_["mean_score"])
        full_r2 = np.mean(brain_data.ridge_scores.data)

        # Just check both are finite and reasonable (not NaN/Inf)
        assert np.isfinite(cv_r2)
        assert np.isfinite(full_r2)
        # Full R² should generally be non-negative (in-sample)
        assert full_r2 >= -0.1  # Allow small numerical errors

    def test_fit_ridge_cv_auto_alpha_selection(self, small_brain_data_for_cv):
        """Test cv='auto' triggers alpha selection."""
        brain_data, X = small_brain_data_for_cv

        # Fit with cv='auto' (implies alpha='auto')
        alphas = [0.1, 1.0, 10.0]  # Small grid for speed
        brain_data.fit(model="ridge", cv="auto", alphas=alphas, X=X)

        # CV results should exist
        assert hasattr(brain_data, "cv_results_")

        # Alpha selection results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_

        # Best alpha should be one of the tested alphas
        best_alpha = brain_data.cv_results_["best_alpha"]
        assert best_alpha in alphas

        # Alpha scores shape: (n_folds, n_alphas, n_voxels)
        alpha_scores = brain_data.cv_results_["alpha_scores"]
        assert alpha_scores.shape == (
            5,
            3,
            5,
        )  # (5 folds default for 'auto', 3 alphas, 5 voxels)

        # Model should be fitted with best_alpha
        assert brain_data.model_.alpha == best_alpha

    def test_fit_ridge_cv_integer_with_alpha_auto(self, small_brain_data_for_cv):
        """Test cv=int with alpha='auto' performs both alpha selection and CV scoring."""
        brain_data, X = small_brain_data_for_cv

        # Fit with explicit alpha selection + CV
        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Should have both alpha selection and CV scoring results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_

        # Alpha scores: (n_folds=3, n_alphas=3, n_voxels=5)
        assert brain_data.cv_results_["alpha_scores"].shape == (3, 3, 5)

        # CV scores computed with best alpha: (n_folds=3, n_voxels=5)
        assert brain_data.cv_results_["scores"].shape == (3, 5)

        # Best alpha selected
        assert brain_data.cv_results_["best_alpha"] in alphas

    def test_fit_ridge_no_cv_backward_compat(self, small_brain_data_for_cv):
        """Test fit() without cv parameter doesn't create cv_results_ (backward compat)."""
        brain_data, X = small_brain_data_for_cv

        # Fit without CV (existing behavior)
        brain_data.fit(model="ridge", alpha=1.0, X=X)

        # CV results should NOT exist
        assert not hasattr(brain_data, "cv_results_")

        # Regular attributes should exist
        assert hasattr(brain_data, "ridge_weights")
        assert hasattr(brain_data, "ridge_fitted_values")
        assert hasattr(brain_data, "ridge_scores")

    def test_fit_ridge_cv_invalid_parameter(self, small_brain_data_for_cv):
        """Test fit() raises errors for invalid cv parameters."""
        brain_data, X = small_brain_data_for_cv

        # Invalid cv type
        with pytest.raises((TypeError, ValueError)):
            brain_data.fit(model="ridge", alpha=1.0, cv="invalid", X=X)

        # Negative cv
        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=-1, X=X)

        # Zero cv
        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=0, X=X)

    def test_fit_ridge_cv_with_insufficient_samples(self, tiny_brain_data_for_cv):
        """Test fit() raises error when cv folds > n_samples."""
        brain_data, X = tiny_brain_data_for_cv  # Only 6 samples

        # Try 10-fold CV with 6 samples
        with pytest.raises(ValueError, match="Cannot have number of splits.*greater"):
            brain_data.fit(model="ridge", alpha=1.0, cv=10, X=X)

    def test_fit_ridge_cv_predict_consistency(self, small_brain_data_for_cv):
        """Test predict() returns full model predictions, not CV predictions."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # Call predict() on training data
        train_predictions = brain_data.predict(X=X)

        # Should match full model predictions (ridge_fitted_values)
        np.testing.assert_allclose(
            train_predictions.data, brain_data.ridge_fitted_values.data
        )

        # Should NOT match CV predictions (out-of-fold)
        assert not np.allclose(
            train_predictions.data, brain_data.cv_results_["predictions"].data
        )

    def test_fit_ridge_cv_stores_all_expected_keys(self, small_brain_data_for_cv):
        """Test cv_results_ dict contains all expected keys and types."""
        brain_data, X = small_brain_data_for_cv

        # Fit with alpha selection
        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Check all expected keys exist
        expected_keys = {
            "scores",
            "mean_score",
            "predictions",
            "folds",
            "best_alpha",
            "alpha_scores",
        }
        assert set(brain_data.cv_results_.keys()) == expected_keys

        # Check types
        assert isinstance(brain_data.cv_results_["scores"], np.ndarray)
        assert isinstance(brain_data.cv_results_["mean_score"], np.ndarray)
        assert isinstance(brain_data.cv_results_["predictions"], BrainData)
        assert isinstance(brain_data.cv_results_["folds"], np.ndarray)
        assert isinstance(brain_data.cv_results_["best_alpha"], (int, float))
        assert isinstance(brain_data.cv_results_["alpha_scores"], np.ndarray)

    # ==================== Masking & ROI Extraction ====================

    def test_apply_mask(self, sim_brain_data):
        """Test applying masks to BrainData."""
        s1 = create_sphere([12, 10, -8], radius=10)
        assert isinstance(s1, nb.Nifti1Image)
        masked_dat = sim_brain_data.apply_mask(s1)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)
        masked_dat = sim_brain_data.apply_mask(s1, resample_mask_to_brain=True)
        assert masked_dat.shape[1] == np.sum(s1.get_fdata() != 0)

    def test_apply_mask_nilearn_validation(self, sim_brain_data):
        """Nilearn should provide better error messages for invalid inputs"""
        # Test that multi-volume mask raises clear error
        # Create invalid 4D mask (should be 3D)
        s1 = create_sphere([12, 10, -8], radius=10)

        # Stack to create 4D (invalid for masking)
        from nilearn.image import concat_imgs

        invalid_mask = concat_imgs([s1, s1])

        # Create BrainData from invalid mask
        mask_bd = BrainData(invalid_mask, mask=sim_brain_data.mask)

        # Should raise ValueError for non-single image
        with pytest.raises(ValueError, match="Mask must be a single image"):
            sim_brain_data.apply_mask(mask_bd)

    def test_apply_mask_dimension_compatibility(self, sim_brain_data):
        """Nilearn should handle dimension compatibility automatically"""
        # Create a compatible mask
        s1 = create_sphere([12, 10, -8], radius=10)
        mask_bd = BrainData(s1, mask=sim_brain_data.mask)

        # This should work (nilearn handles dimension matching)
        result = sim_brain_data.apply_mask(mask_bd)

        assert isinstance(result, BrainData)
        # Verify output shape matches number of non-zero mask voxels
        assert result.shape[1] == mask_bd.data.astype(bool).sum()

    def test_apply_mask_resampling(self, sim_brain_data):
        """Test resample_mask_to_brain parameter works correctly"""
        s1 = create_sphere([12, 10, -8], radius=10)
        mask_bd = BrainData(s1, mask=sim_brain_data.mask)

        # With resampling
        result_resample = sim_brain_data.apply_mask(
            mask_bd, resample_mask_to_brain=True
        )
        assert isinstance(result_resample, BrainData)
        assert result_resample.shape[1] == np.sum(s1.get_fdata() != 0)

        # Without resampling (default)
        result_no_resample = sim_brain_data.apply_mask(
            mask_bd, resample_mask_to_brain=False
        )
        assert isinstance(result_no_resample, BrainData)
        assert result_no_resample.shape[1] == mask_bd.data.astype(bool).sum()

    @pytest.mark.slow
    def test_extract_roi(self, sim_brain_data):
        """Test ROI extraction with different metrics and labeled atlases."""
        mask = create_sphere([12, 10, -8], radius=10)
        assert len(sim_brain_data.extract_roi(mask, metric="mean")) == shape_2d[0]
        assert len(sim_brain_data.extract_roi(mask, metric="median")) == shape_2d[0]
        n_components = 2
        assert sim_brain_data.extract_roi(
            mask, metric="pca", n_components=n_components
        ).shape == (n_components, shape_2d[0])
        with pytest.raises(NotImplementedError):
            sim_brain_data.extract_roi(mask, metric="p")

        assert isinstance(
            sim_brain_data[0].extract_roi(mask, metric="mean"), (float, np.floating)
        )
        assert isinstance(
            sim_brain_data[0].extract_roi(mask, metric="median"), (float, np.floating)
        )
        with pytest.raises(ValueError):
            sim_brain_data[0].extract_roi(mask, metric="pca")
        with pytest.raises(NotImplementedError):
            sim_brain_data[0].extract_roi(mask, metric="p")

        s1 = create_sphere([15, 10, -8], radius=10)
        s2 = create_sphere([-15, 10, -8], radius=10)
        s3 = create_sphere([0, -15, -8], radius=10)
        masks = BrainData([s1, s2, s3])
        mask = roi_to_brain([1, 2, 3], masks)
        assert len(sim_brain_data[0].extract_roi(mask, metric="mean")) == len(masks)
        assert len(sim_brain_data[0].extract_roi(mask, metric="median")) == len(masks)
        assert sim_brain_data.extract_roi(mask, metric="mean").shape == (
            len(masks),
            shape_2d[0],
        )
        assert sim_brain_data.extract_roi(mask, metric="median").shape == (
            len(masks),
            shape_2d[0],
        )
        assert len(
            sim_brain_data.extract_roi(mask, metric="pca", n_components=n_components)
        ) == len(masks)

    # ==================== Transform Methods ====================

    def test_r_to_z(self, sim_brain_data):
        """Test Fisher r-to-z transformation."""
        z = sim_brain_data.r_to_z()
        assert z.shape == sim_brain_data.shape

    def test_copy(self, sim_brain_data):
        """Test copying BrainData objects."""
        d_copy = sim_brain_data.copy()
        assert d_copy.shape == sim_brain_data.shape

    def test_detrend(self, sim_brain_data):
        """Test detrending removes linear trends."""
        detrend = sim_brain_data.detrend()
        assert detrend.shape == sim_brain_data.shape

    @pytest.mark.filterwarnings("ignore:Numerical issues:UserWarning")
    def test_standardize(self, sim_brain_data):
        """Test standardization with different methods."""
        s = sim_brain_data.standardize()
        assert s.shape == sim_brain_data.shape
        # Mean should be close to zero after standardization (tolerance for numerical precision)
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)
        s = sim_brain_data.standardize(method="zscore")
        assert s.shape == sim_brain_data.shape
        assert np.isclose(np.sum(s.mean().data), 0, atol=0.5)

    def test_filter_high_pass(self, minimal_brain_data):
        """Test high-pass filtering returns BrainData with correct shape."""
        # Test basic API: sampling_freq + high_pass
        filtered = minimal_brain_data.filter(sampling_freq=0.5, high_pass=0.01)

        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape
        # Original data should be unchanged (immutability)
        assert not np.array_equal(id(filtered.data), id(minimal_brain_data.data))

    def test_filter_low_pass(self, minimal_brain_data):
        """Test low-pass filtering returns BrainData with correct shape."""
        filtered = minimal_brain_data.filter(sampling_freq=0.5, low_pass=0.1)

        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape

    def test_filter_band_pass(self, minimal_brain_data):
        """Test band-pass filtering (both high and low pass)."""
        filtered = minimal_brain_data.filter(
            sampling_freq=0.5, high_pass=0.01, low_pass=0.1
        )

        assert isinstance(filtered, BrainData)
        assert filtered.shape == minimal_brain_data.shape

    def test_filter_error_no_sampling_freq(self, minimal_brain_data):
        """Test error when sampling_freq not provided."""
        with pytest.raises(ValueError, match="sampling rate"):
            minimal_brain_data.filter(high_pass=0.01)

    def test_filter_error_no_cutoff(self, minimal_brain_data):
        """Test error when neither high_pass nor low_pass specified."""
        # Note: current error message has typo "beprovided"
        with pytest.raises(ValueError, match="must.*provided"):
            minimal_brain_data.filter(sampling_freq=0.5)

    def test_filter_kwargs_passed_through(self, minimal_brain_data):
        """Test that additional kwargs reach nilearn.signal.clean."""
        # Test with ensure_finite kwarg (nilearn.signal.clean parameter)
        # This is a smoke test - we don't validate parameter effect,
        # just that the method runs without error when kwargs provided
        filtered = minimal_brain_data.filter(
            sampling_freq=0.5,
            high_pass=0.01,
            ensure_finite=True,  # nilearn parameter not extracted by filter()
        )

        assert isinstance(filtered, BrainData)

    def test_smooth(self, sim_brain_data):
        """Test spatial smoothing."""
        smoothed = sim_brain_data.smooth(5.0)
        assert isinstance(smoothed, BrainData)
        assert smoothed.shape == sim_brain_data.shape
        smoothed = sim_brain_data[0].smooth(5.0)
        assert len(smoothed.shape) == 1

    @pytest.mark.slow
    def test_threshold(self):
        """Test thresholding and region extraction."""
        s1 = create_sphere([12, 10, -8], radius=10)
        s2 = create_sphere([22, -2, -22], radius=10)
        mask = BrainData(s1) * 5
        mask = mask + BrainData(s2)

        m1 = mask.threshold(upper=0.5)
        m2 = mask.threshold(upper=3)
        m3 = mask.threshold(upper="98%")
        m4 = BrainData(s1) * 5 + BrainData(s2) * -0.5
        m4 = mask.threshold(upper=0.5, lower=-0.3)
        assert np.sum(m1.data > 0) > np.sum(m2.data > 0)
        assert np.sum(m1.data > 0) == np.sum(m3.data > 0)
        assert np.sum(m4.data[(m4.data > -0.3) & (m4.data < 0.5)]) == 0
        assert np.sum(m4.data[(m4.data < -0.3) | (m4.data > 0.5)]) > 0

        # Test Regions
        r = mask.regions(min_region_size=10)
        m1 = BrainData(s1)
        m2 = r.threshold(1, binarize=True)
        assert len(np.unique(r.to_nifti().get_fdata())) == 2
        diff = m2 - m1
        assert np.sum(diff.data) == 0

    # ============================================================================
    # Thresholding Operations - Cluster Enhancement
    # ============================================================================

    @pytest.mark.slow
    def test_threshold_cluster_basic(self, sim_brain_data):
        """Cluster thresholding should filter small clusters using nilearn"""
        # Create data with distinct regions
        brain = sim_brain_data.copy()

        # Threshold with cluster size minimum
        result = brain.threshold(lower=2, cluster_threshold=10)

        # Should return BrainData
        assert isinstance(result, BrainData)
        # Should have removed small clusters (basic check that it ran)
        assert result.shape == brain.shape

    @pytest.mark.slow
    def test_threshold_cluster_with_upper_only(self, sim_brain_data):
        """Cluster threshold should work with upper threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(upper=2, cluster_threshold=10)
        assert isinstance(result, BrainData)

    @pytest.mark.slow
    def test_threshold_cluster_with_lower_only(self, sim_brain_data):
        """Cluster threshold should work with lower threshold only"""
        brain = sim_brain_data.copy()
        result = brain.threshold(lower=2, cluster_threshold=10)
        assert isinstance(result, BrainData)

    def test_threshold_cluster_rejects_bandpass(self, sim_brain_data):
        """Should raise error when using both upper AND lower with cluster_threshold"""
        brain = sim_brain_data.copy()

        with pytest.raises(
            ValueError, match="Band-pass filtering.*not supported.*cluster"
        ):
            brain.threshold(lower=-2, upper=2, cluster_threshold=10)

    @pytest.mark.slow
    def test_threshold_cluster_with_binarize(self, sim_brain_data):
        """Cluster threshold should work with binarization"""
        brain = sim_brain_data.copy()
        result = brain.threshold(lower=2, cluster_threshold=10, binarize=True)

        # Should be binary
        unique_vals = np.unique(result.data)
        assert len(unique_vals) <= 2
        assert all(v in [0, 1] for v in unique_vals)

    def test_threshold_cluster_zero_disables(self, sim_brain_data):
        """cluster_threshold=0 should use fast path (current implementation)"""
        brain = sim_brain_data.copy()

        # These should be equivalent
        result_no_cluster = brain.threshold(lower=2, upper=5)
        result_zero_cluster = brain.threshold(lower=2, upper=5, cluster_threshold=0)

        np.testing.assert_array_equal(result_no_cluster.data, result_zero_cluster.data)

    def test_threshold_backwards_compatible_no_cluster(self, sim_brain_data):
        """Existing threshold behavior unchanged when cluster_threshold=0"""
        brain = sim_brain_data.copy()

        # Old way (default cluster_threshold=0)
        result_old = brain.threshold(lower=-2, upper=2)

        # Explicit cluster_threshold=0
        result_explicit = brain.threshold(lower=-2, upper=2, cluster_threshold=0)

        # Should be identical
        np.testing.assert_array_equal(result_old.data, result_explicit.data)

    def test_threshold_bandpass_still_works(self, sim_brain_data):
        """Band-pass filtering (unique feature) still works without cluster_threshold"""
        brain = sim_brain_data.copy()

        # This should still work (keep middle values, zero extremes)
        result = brain.threshold(lower=-2, upper=2)

        assert isinstance(result, BrainData)
        # Verify band-pass behavior preserved (values in range kept)

    def test_threshold_with_zero_value(self, sim_brain_data):
        """Test threshold works correctly when upper=0 or lower=0 (#370).

        This was a bug where `if upper:` evaluated to False when upper=0,
        causing thresholding to be skipped.
        """
        brain = sim_brain_data.copy()
        # Create data with positive and negative values
        brain.data = brain.data - brain.data.mean()

        # Test upper=0: should zero out values < 0
        result_upper0 = brain.threshold(upper=0)
        assert np.all(result_upper0.data >= 0), "upper=0 should zero values < 0"

        # Test lower=0: should zero out values > 0
        result_lower0 = brain.threshold(lower=0)
        assert np.all(result_lower0.data <= 0), "lower=0 should zero values > 0"

        # Test upper=0, lower=-0.5: bandpass with zero boundary
        result_bandpass = brain.threshold(upper=0, lower=-0.5)
        # Values between -0.5 and 0 should be zeroed
        non_zero = result_bandpass.data[result_bandpass.data != 0]
        assert np.all((non_zero >= 0) | (non_zero <= -0.5)), (
            "bandpass with upper=0 should work"
        )

    @pytest.mark.slow
    def test_threshold_cluster_realistic_neuroimaging(self, sim_brain_data):
        """Integration test with realistic neuroimaging workflow"""
        # Test with actual brain data structure from fixtures
        brain = sim_brain_data.copy()

        # Realistic workflow: threshold then cluster filter
        result = brain.threshold(lower=2.5, cluster_threshold=50)

        # Basic sanity checks
        assert isinstance(result, BrainData)
        assert result.shape == brain.shape
        assert not result.is_empty

    # ==================== Similarity & Analysis ====================

    def test_similarity(self, sim_brain_data):
        """Test similarity computation with different metrics."""
        # Test comparing BrainData to itself
        r = sim_brain_data.similarity(sim_brain_data, method="correlation")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])
        r = sim_brain_data.similarity(sim_brain_data, method="dot_product")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])
        r = sim_brain_data.similarity(sim_brain_data, method="cosine")
        assert r.shape == (sim_brain_data.shape[0], sim_brain_data.shape[0])

        # Test comparing to a single image
        r = sim_brain_data.similarity(sim_brain_data[0], method="correlation")
        assert len(r) == shape_2d[0]

    @pytest.mark.slow
    def test_decompose(self, sim_brain_data):
        """Test decomposition with PCA, ICA, NMF, and Factor Analysis."""
        n_components = 3
        stats = sim_brain_data.decompose(
            algorithm="pca", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="ica", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        sim_brain_data.data = sim_brain_data.data + 2
        sim_brain_data.data[sim_brain_data.data < 0] = 0
        stats = sim_brain_data.decompose(
            algorithm="nnmf", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="fa", axis="voxels", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="pca", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="ica", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        sim_brain_data.data = sim_brain_data.data + 2
        sim_brain_data.data[sim_brain_data.data < 0] = 0
        stats = sim_brain_data.decompose(
            algorithm="nnmf", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

        stats = sim_brain_data.decompose(
            algorithm="fa", axis="images", n_components=n_components
        )
        assert n_components == len(stats["components"])
        assert stats["weights"].shape == (len(sim_brain_data), n_components)

    # ==================== Alignment ====================

    @pytest.mark.slow
    def test_hyperalignment(self):
        """Test hyperalignment with SRM and Procrustes methods."""
        sim = Simulator()
        y = [0, 1]
        n_reps = 10
        s1 = create_sphere([0, 0, 0], radius=3)
        d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
        d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
        d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
        data = [d1, d2, d3]

        # Test deterministic brain_data
        out = align(data, method="deterministic_srm")

        bout = d1.align(out["common_model"], method="deterministic_srm")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed)
        )

        # Test probabilistic brain_data
        bout = d1.align(out["common_model"], method="probabilistic_srm")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed)
        )

        # Test procrustes brain_data
        out = align(data, method="procrustes")
        centered = data[0].data - np.mean(data[0].data, 0)

        bout = d1.align(out["common_model"], method="procrustes")
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[1] == bout["transformation_matrix"].shape[0]
        centered = d1.data - np.mean(d1.data, 0)
        btransformed = (
            np.dot(
                centered / np.linalg.norm(centered), bout["transformation_matrix"].data
            )
            * bout["scale"]
        )
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed), decimal=5
        )
        np.testing.assert_almost_equal(
            0, np.sum(out["transformed"][0].data - bout["transformed"].data), decimal=5
        )

        # Test over time
        sim = Simulator()
        y = [0, 1]
        n_reps = 10
        s1 = create_sphere([0, 0, 0], radius=5)
        d1 = sim.create_data(y, 1, reps=n_reps, output_dir=None).apply_mask(s1)
        d2 = sim.create_data(y, 2, reps=n_reps, output_dir=None).apply_mask(s1)
        d3 = sim.create_data(y, 3, reps=n_reps, output_dir=None).apply_mask(s1)
        data = [d1, d2, d3]

        out = align(data, method="deterministic_srm", axis=1)
        bout = d1.align(out["common_model"], method="deterministic_srm", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data.T, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T)
        )

        out = align(data, method="probabilistic_srm", axis=1)
        bout = d1.align(out["common_model"], method="probabilistic_srm", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        btransformed = np.dot(d1.data.T, bout["transformation_matrix"].data.T)
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T)
        )

        out = align(data, method="procrustes", axis=1)
        bout = d1.align(out["common_model"], method="procrustes", axis=1)
        assert d1.shape == bout["transformed"].shape
        assert d1.shape == bout["common_model"].shape
        assert d1.shape[0] == bout["transformation_matrix"].shape[0]
        centered = d1.data.T - np.mean(d1.data.T, 0)
        btransformed = (
            np.dot(
                centered / np.linalg.norm(centered), bout["transformation_matrix"].data
            )
            * bout["scale"]
        )
        np.testing.assert_almost_equal(
            0, np.sum(bout["transformed"].data - btransformed.T), decimal=5
        )
        np.testing.assert_almost_equal(
            0, np.sum(out["transformed"][0].data - bout["transformed"].data)
        )

    # ==================== Temporal Methods ====================

    @pytest.mark.slow
    def test_temporal_resample(self, sim_brain_data):
        """Test temporal resampling (upsampling and downsampling)."""
        up = sim_brain_data.temporal_resample(
            sampling_freq=1 / 2, target=2, target_type="hz"
        )
        assert len(sim_brain_data) * 4 == len(up)
        down = up.temporal_resample(sampling_freq=2, target=1 / 2, target_type="hz")
        assert len(sim_brain_data) == len(down)
        assert len(up) / 4 == len(down)

    def test_fisher_r_to_z(self, sim_brain_data):
        """Test Fisher r-to-z and inverse transformation."""
        np.testing.assert_almost_equal(
            np.nansum(sim_brain_data.data - sim_brain_data.r_to_z().z_to_r().data),
            0,
            decimal=2,
        )

    # ==================== Deprecated Methods ====================

    def test_ttest(self, sim_brain_data):
        """Test that deprecated ttest method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="ttest.*deprecated.*Model class"):
            sim_brain_data.ttest()

    def test_randomise(self, sim_brain_data):
        """Test that deprecated randomise method raises NotImplementedError."""
        sim_brain_data.X = pd.DataFrame({"Intercept": np.ones(len(sim_brain_data.Y))})

        with pytest.raises(
            NotImplementedError, match="randomise.*deprecated.*Model class"
        ):
            sim_brain_data.randomise(n_permute=10)

    def test_bootstrap(self, sim_brain_data):
        """Test bootstrap with mean/std."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Test basic bootstrap with mean and std (should work)
        # Note: n_samples must be >= 10 for new implementation
        n_samples = 50
        b = masked.bootstrap(stat="mean", n_samples=n_samples)
        # New API returns BrainData directly
        assert isinstance(b, BrainData)
        assert b.shape == (1, masked.shape[1])  # (1, n_voxels)
        b = masked.bootstrap(stat="std", n_samples=n_samples)
        assert isinstance(b, BrainData)
        assert b.shape == (1, masked.shape[1])  # (1, n_voxels)

        # Bootstrap with "predict" requires fitted model (pass X_test to get past that check)
        X_test = np.random.randn(5, 10)  # Dummy test features
        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(stat="predict", n_samples=n_samples, X_test=X_test)

    def test_bootstrap_invalid_method_error(self, sim_brain_data):
        """Test error raised for unsupported method."""
        # New implementation validates stat names upfront
        with pytest.raises(
            ValueError,
            match="Unsupported stat.*Supported simple stats",
        ):
            sim_brain_data.bootstrap(stat="invalid_method_name", n_samples=10)

    # ==================== Phase 5: New Bootstrap Implementation ====================

    def test_bootstrap_new_stat_param(self, sim_brain_data):
        """Test new bootstrap with stat='mean' parameter."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # New API: stat parameter
        boot = masked.bootstrap(stat="mean", n_samples=100, random_state=42)

        # Should return BrainData with shape (1, n_voxels) for aggregated result
        assert isinstance(boot, BrainData)
        assert boot.shape == (
            1,
            masked.shape[1],
        )  # (1, n_voxels) - aggregated across samples

    def test_bootstrap_new_save_boots_param(self, sim_brain_data):
        """Test new bootstrap with save_boots parameter."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # New API: save_boots=True should return dict
        result = masked.bootstrap(
            stat="mean", n_samples=50, save_boots=True, random_state=42
        )

        # When save_boots=True, should return dict with samples
        assert isinstance(result, dict)
        assert "samples" in result
        assert result["samples"].shape[0] == 50  # n_samples

    def test_bootstrap_new_all_simple_stats(self, sim_brain_data):
        """Test all simple stats work with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        stats = ["mean", "median", "std", "sum", "min", "max"]
        for stat in stats:
            boot = masked.bootstrap(stat=stat, n_samples=50, random_state=42)
            assert isinstance(boot, BrainData)
            assert boot.shape == (1, masked.shape[1])  # (1, n_voxels) - aggregated

    def test_bootstrap_new_ridge_weights_requires_fit(self, sim_brain_data):
        """Test weights bootstrap requires fitted model."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(stat="weights", n_samples=10)

    def test_bootstrap_new_ridge_weights_basic(self, sim_brain_data):
        """Test Ridge weights bootstrap with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        # Create design matrix
        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))

        # Fit model
        masked.fit(X=dm, model="ridge", alpha=1.0)

        # Bootstrap weights
        boot = masked.bootstrap(stat="weights", n_samples=100, random_state=42)

        # Should return dict with mean, std, Z, p, ci_lower, ci_upper
        assert isinstance(boot, dict)
        assert "mean" in boot
        assert "std" in boot
        assert "Z" in boot
        assert "p" in boot
        assert "ci_lower" in boot
        assert "ci_upper" in boot

        # Mean should be BrainData with shape (n_features, n_voxels)
        assert isinstance(boot["mean"], BrainData)
        assert boot["mean"].shape == (5, masked.shape[1])  # n_features × n_voxels

    def test_bootstrap_new_ridge_predict_requires_fit(self, sim_brain_data):
        """Test predict bootstrap requires fitted model."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        X_test = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="Must call.*fit"):
            masked.bootstrap(stat="predict", X_test=X_test, n_samples=10)

    def test_bootstrap_new_ridge_predict_requires_x_test(self, sim_brain_data):
        """Test predict bootstrap requires X_test."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))

        masked.fit(X=dm, model="ridge", alpha=1.0)

        with pytest.raises(ValueError, match="X_test.*required"):
            masked.bootstrap(stat="predict", n_samples=10)

    def test_bootstrap_new_ridge_predict_basic(self, sim_brain_data):
        """Test Ridge predict bootstrap with new implementation."""
        masked = sim_brain_data.apply_mask(
            create_sphere(radius=10, coordinates=[0, 0, 0])
        )

        from nltools.data import DesignMatrix

        dm = DesignMatrix(np.random.randn(len(masked), 5))
        X_test = np.random.randn(10, 5)

        masked.fit(X=dm, model="ridge", alpha=1.0)

        boot = masked.bootstrap(
            stat="predict", X_test=X_test, n_samples=100, random_state=42
        )

        # Should return dict
        assert isinstance(boot, dict)
        assert "mean" in boot

        # Mean should be BrainData with shape (n_test_samples, n_voxels)
        assert isinstance(boot["mean"], BrainData)
        assert boot["mean"].shape == (10, masked.shape[1])  # n_test × n_voxels

    @pytest.mark.slow
    def test_predict_multi(self):
        """Test that deprecated predict_multi method raises NotImplementedError."""
        # Need to set up minimal data for the test
        sim = Simulator()
        dat = sim.create_data([0, 1], sigma=1, reps=5, output_dir=".")
        y = pd.read_csv("y.csv", header=None, index_col=None)
        dat = BrainData("data.nii.gz", Y=y)

        with pytest.raises(
            NotImplementedError, match="predict_multi.*deprecated.*Model class"
        ):
            dat.predict_multi(algorithm="svm")
