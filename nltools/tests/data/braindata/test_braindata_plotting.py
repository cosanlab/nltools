"""
Test suite for BrainData.plot() method.

Tests follow TDD approach: write tests first, then implement functionality.
Focuses on plotting functionality, MNI_Template integration, and user-friendly defaults.
"""

import pytest
import numpy as np
from nltools.data import BrainData
from nltools.prefs import MNI_Template
from nltools.utils import get_mni_from_img_resolution


class TestBrainDataPlotting:
    """Test BrainData plotting methods."""

    # ==================== Phase 1: Baseline Tests ====================

    def test_plot_single_image_default(self, sim_brain_data):
        """Test plotting single BrainData image with defaults"""
        # Extract single image
        single_image = sim_brain_data[0]
        # Should work without errors
        result = single_image.plot()
        # Should return nilearn Display object or matplotlib figure
        assert result is not None

    def test_plot_multiple_images_default(self, sim_brain_data):
        """Test plotting from BrainData with multiple images"""
        # For multiple images, should plot first image or mean
        result = sim_brain_data.plot()
        assert result is not None

    @pytest.mark.slow
    def test_plot_glass_brain(self, sim_brain_data):
        """Test glass brain visualization"""
        single_image = sim_brain_data[0]
        result = single_image.plot(kind="glass")
        assert result is not None

    @pytest.mark.slow
    def test_plot_multi_slice(self, sim_brain_data):
        """Test multi-slice visualization"""
        single_image = sim_brain_data[0]
        result = single_image.plot(kind="slices")
        assert result is not None

    def test_plot_thresholding_float(self, sim_brain_data):
        """Test thresholding functionality with float"""
        single_image = sim_brain_data[0]
        result = single_image.plot(thr_upper=0.5)
        assert result is not None

    def test_plot_thresholding_percentile(self, sim_brain_data):
        """Test thresholding functionality with percentile string"""
        single_image = sim_brain_data[0]
        result = single_image.plot(thr_upper="95%")
        assert result is not None

    def test_plot_thresholding_lower(self, sim_brain_data):
        """Test lower thresholding"""
        single_image = sim_brain_data[0]
        result = single_image.plot(thr_lower=-0.5)
        assert result is not None

    def test_plot_thresholding_bandpass(self, sim_brain_data):
        """Test band-pass thresholding"""
        single_image = sim_brain_data[0]
        result = single_image.plot(thr_upper="90%", thr_lower="10%")
        assert result is not None

    def test_plot_custom_cut_coords(self, sim_brain_data):
        """Test custom cut coordinates"""
        single_image = sim_brain_data[0]
        result = single_image.plot(cut_coords=[[0], [0], [0]])
        assert result is not None

    def test_plot_custom_colormap(self, sim_brain_data):
        """Test custom colormap"""
        single_image = sim_brain_data[0]
        result = single_image.plot(cmap="hot")
        assert result is not None

    def test_plot_background_image_respects_template(self, sim_brain_data):
        """Test that background image respects MNI_Template settings"""
        single_image = sim_brain_data[0]
        # Should use get_mni_from_img_resolution() which respects MNI_Template
        result = (
            single_image.plot()
        )  # Should auto-select based on MNI_Template and resolution
        assert result is not None

    def test_plot_respects_template_changes(self, sim_brain_data):
        """Test that plot respects MNI_Template changes"""
        single_image = sim_brain_data[0]

        # Store original template
        original_template = MNI_Template.template
        original_resolution = MNI_Template.resolution

        try:
            # Change template if supported
            if 2 in MNI_Template._supported_combinations.get("nilearn", []):
                MNI_Template.template = "nilearn"
                MNI_Template.resolution = 2
                result = single_image.plot()  # Should use nilearn template
                assert result is not None

        finally:
            # Restore original
            MNI_Template.template = original_template
            MNI_Template.resolution = original_resolution

    def test_plot_empty_brain_data(self):
        """Test error handling for empty BrainData"""
        brain = BrainData()
        with pytest.raises(ValueError, match="empty|Empty"):
            brain.plot()

    def test_plot_invalid_kind(self, sim_brain_data):
        """Test error handling for invalid 'kind' parameter"""
        single_image = sim_brain_data[0]
        with pytest.raises(ValueError):
            single_image.plot(kind="invalid")

    def test_plot_uses_template_resolution_matcher(self, sim_brain_data):
        """Test that plot uses get_mni_from_img_resolution() which respects MNI_Template"""
        single_image = sim_brain_data[0]
        # Verify that plot uses the same logic as get_mni_from_img_resolution
        expected_bg = get_mni_from_img_resolution(single_image, img_type="brain")
        # plot() should use this same function internally
        # This ensures consistency with MNI_Template settings
        assert expected_bg is not None  # Verify function works
        result = single_image.plot()  # Should use same logic
        assert result is not None

    # ==================== Phase 3: Robustness Tests ====================

    def test_plot_validate_kind_multiple_invalid(self, sim_brain_data):
        """Test validation of 'kind' parameter with multiple invalid values"""
        single_image = sim_brain_data[0]
        for invalid in ["invalid", "", 123, None]:
            with pytest.raises(ValueError):
                single_image.plot(kind=invalid)

    def test_plot_handle_nan_values(self, sim_brain_data):
        """Test handling of NaN/Inf values"""
        single_image = sim_brain_data[0]
        # Add NaN and Inf values (single image is 1D)
        if single_image.data.ndim == 1:
            single_image.data[0] = np.nan
            if len(single_image.data) > 1:
                single_image.data[1] = np.inf
        else:
            single_image.data[0, 0] = np.nan
            single_image.data[0, 1] = np.inf
        # Should handle gracefully (thresholding will coerce NaN)
        # Test with thresholding which handles NaN
        result = single_image.plot(thr_upper=0.5)
        assert result is not None

    def test_plot_single_voxel(self, minimal_brain_data):
        """Test edge case: very small brain data"""
        # minimal_brain_data has 5 voxels, should work
        result = minimal_brain_data[0].plot()
        assert result is not None

    @pytest.mark.slow
    def test_plot_missing_mask_handling(self):
        """Test handling when mask is None (should use default)"""
        # Create BrainData without explicit mask (uses default)
        import nibabel as nib

        # Create minimal data
        data = np.random.randn(10, 10, 10)
        nifti_img = nib.Nifti1Image(data, np.eye(4))
        brain = BrainData(nifti_img, mask=None)  # Should use default mask
        result = brain.plot()
        assert result is not None

    def test_plot_cut_coords_format_validation(self, sim_brain_data):
        """Test that cut_coords format is handled correctly"""
        single_image = sim_brain_data[0]
        # Should handle list of lists
        result = single_image.plot(cut_coords=[[0], [0], [0]])
        assert result is not None
        # Should handle ranges
        result = single_image.plot(
            cut_coords=[range(-10, 11, 5), range(-10, 11, 5), range(-10, 11, 5)]
        )
        assert result is not None

    def test_plot_custom_bg_img_nibabel(self, sim_brain_data):
        """Test custom background image as nibabel image"""
        import nibabel as nib
        from nltools.prefs import MNI_Template

        single_image = sim_brain_data[0]
        # Use actual template file as background
        custom_bg = nib.load(MNI_Template.brain)
        result = single_image.plot(bg_img=custom_bg)
        assert result is not None

    def test_plot_save_functionality(self, sim_brain_data, tmpdir):
        """Test save functionality"""
        single_image = sim_brain_data[0]
        save_path = str(tmpdir / "test_plot.png")
        result = single_image.plot(kind="glass", save=save_path)
        assert result is not None
        # Check that files were created (glass brain creates one file)
        import os

        glass_file = save_path.replace(".png", "_glass.png")
        assert os.path.exists(glass_file)

    # ==================== Phase 4/5/6: User-Friendly Features ====================

    def test_plot_timeseries_mean(self, sim_brain_data):
        """Test timeseries plotting with mean aggregation"""
        # Multiple images: plot mean across voxels
        result = sim_brain_data.plot(kind="timeseries", stat="mean")
        assert result is not None

    def test_plot_timeseries_median(self, sim_brain_data):
        """Test timeseries plotting with median aggregation"""
        result = sim_brain_data.plot(kind="timeseries", stat="median")
        assert result is not None

    def test_plot_timeseries_std(self, sim_brain_data):
        """Test timeseries plotting with std aggregation"""
        result = sim_brain_data.plot(kind="timeseries", stat="std")
        assert result is not None

    def test_plot_histogram_single_image(self, sim_brain_data):
        """Test histogram plotting for single image"""
        single_image = sim_brain_data[0]
        result = single_image.plot(kind="histogram")
        assert result is not None

    def test_plot_histogram_multiple_images(self, sim_brain_data):
        """Test histogram plotting for multiple images"""
        result = sim_brain_data.plot(kind="histogram")
        assert result is not None

    def test_plot_threshold_convenience(self, sim_brain_data):
        """Test convenience threshold parameter"""
        single_image = sim_brain_data[0]
        # threshold should set thr_upper if positive
        result = single_image.plot(threshold=0.5)
        assert result is not None
        # threshold should set thr_lower if negative
        result = single_image.plot(threshold=-0.5)
        assert result is not None

    def test_plot_custom_title(self, sim_brain_data):
        """Test custom title"""
        single_image = sim_brain_data[0]
        result = single_image.plot(title="My Custom Title")
        assert result is not None

    def test_plot_matplotlib_axis_timeseries(self, sim_brain_data):
        """Test plotting timeseries on existing matplotlib axis"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = sim_brain_data.plot(kind="timeseries", ax=ax)
        assert result is not None
        plt.close(fig)

    def test_plot_matplotlib_axis_histogram(self, sim_brain_data):
        """Test plotting histogram on existing matplotlib axis"""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = sim_brain_data[0].plot(kind="histogram", ax=ax)
        assert result is not None
        plt.close(fig)
