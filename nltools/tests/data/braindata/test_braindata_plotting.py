"""
Test suite for BrainData.plot() method.

Tests follow TDD approach: write tests first, then implement functionality.
Focuses on plotting functionality, MNI_Template integration, and user-friendly defaults.
"""

import pytest
import numpy as np
from nltools.data import BrainData
from nltools.prefs import MNI_Template


class TestBrainDataPlotting:
    """Test BrainData plotting methods."""

    # ==================== Phase 1: Baseline Tests ====================

    def test_plot_single_image_default(self, minimal_brain_data):
        """Test plotting single BrainData image with defaults"""
        result = minimal_brain_data[0].plot()
        assert result is not None

    def test_plot_multiple_images_default(self, minimal_brain_data):
        """Test plotting from BrainData with multiple images"""
        result = minimal_brain_data.plot()
        assert result is not None

    @pytest.mark.slow
    def test_plot_glass_brain(self, minimal_brain_data):
        """Test glass brain visualization"""
        result = minimal_brain_data[0].plot(kind="glass")
        assert result is not None

    @pytest.mark.slow
    def test_plot_multi_slice(self, minimal_brain_data):
        """Test multi-slice visualization"""
        result = minimal_brain_data[0].plot(kind="slices")
        assert result is not None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"thr_upper": 0.5},
            {"thr_upper": "95%"},
            {"thr_lower": -0.5},
            {"thr_upper": "90%", "thr_lower": "10%"},
        ],
        ids=["float", "percentile", "lower", "bandpass"],
    )
    def test_plot_thresholding(self, minimal_brain_data, kwargs):
        """Test thresholding functionality with various inputs."""
        result = minimal_brain_data[0].plot(**kwargs)
        assert result is not None

    def test_plot_custom_cut_coords(self, minimal_brain_data):
        """Test custom cut coordinates"""
        result = minimal_brain_data[0].plot(cut_coords=[[0], [0], [0]])
        assert result is not None

    def test_plot_custom_colormap(self, minimal_brain_data):
        """Test custom colormap"""
        result = minimal_brain_data[0].plot(cmap="hot")
        assert result is not None

    def test_plot_respects_template_changes(self, minimal_brain_data):
        """Test that plot respects MNI_Template changes"""
        single_image = minimal_brain_data[0]

        original_template = MNI_Template.template
        original_resolution = MNI_Template.resolution

        try:
            if 2 in MNI_Template._supported_combinations.get("nilearn", []):
                MNI_Template.template = "nilearn"
                MNI_Template.resolution = 2
                result = single_image.plot()
                assert result is not None
        finally:
            MNI_Template.template = original_template
            MNI_Template.resolution = original_resolution

    def test_plot_empty_brain_data(self):
        """Test error handling for empty BrainData"""
        brain = BrainData()
        with pytest.raises(ValueError, match="empty|Empty"):
            brain.plot()

    @pytest.mark.parametrize("kind", ["invalid", "", 123, None])
    def test_plot_invalid_kind(self, minimal_brain_data, kind):
        """Test error handling for invalid 'kind' parameter"""
        with pytest.raises(ValueError):
            minimal_brain_data[0].plot(kind=kind)

    def test_plot_handle_nan_values(self, minimal_brain_data):
        """Test handling of NaN/Inf values"""
        single_image = minimal_brain_data[0].copy()
        if single_image.data.ndim == 1:
            single_image.data[0] = np.nan
            if len(single_image.data) > 1:
                single_image.data[1] = np.inf
        else:
            single_image.data[0, 0] = np.nan
            single_image.data[0, 1] = np.inf
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

    def test_plot_cut_coords_format_validation(self, minimal_brain_data):
        """Test that cut_coords format is handled correctly"""
        single_image = minimal_brain_data[0]
        result = single_image.plot(cut_coords=[[0], [0], [0]])
        assert result is not None
        result = single_image.plot(
            cut_coords=[range(-10, 11, 5), range(-10, 11, 5), range(-10, 11, 5)]
        )
        assert result is not None

    def test_plot_custom_bg_img_nibabel(self, minimal_brain_data):
        """Test custom background image as nibabel image"""
        import nibabel as nib
        from nltools.prefs import MNI_Template

        single_image = minimal_brain_data[0]
        custom_bg = nib.load(MNI_Template.brain)
        result = single_image.plot(bg_img=custom_bg)
        assert result is not None

    def test_plot_save_functionality(self, minimal_brain_data, tmpdir):
        """Test save functionality"""
        single_image = minimal_brain_data[0]
        save_path = str(tmpdir / "test_plot.png")
        result = single_image.plot(kind="glass", save=save_path)
        assert result is not None
        import os

        glass_file = save_path.replace(".png", "_glass.png")
        assert os.path.exists(glass_file)

    # ==================== Phase 4/5/6: User-Friendly Features ====================

    @pytest.mark.parametrize("stat", ["mean", "median", "std"])
    def test_plot_timeseries(self, minimal_brain_data, stat):
        """Test timeseries plotting with various aggregation stats."""
        result = minimal_brain_data.plot(kind="timeseries", stat=stat)
        assert result is not None

    def test_plot_histogram(self, minimal_brain_data):
        """Test histogram plotting for single and multiple images."""
        result = minimal_brain_data[0].plot(kind="histogram")
        assert result is not None
        result = minimal_brain_data.plot(kind="histogram")
        assert result is not None

    def test_plot_threshold_convenience(self, minimal_brain_data):
        """Test convenience threshold parameter"""
        result = minimal_brain_data[0].plot(threshold=0.5)
        assert result is not None
        result = minimal_brain_data[0].plot(threshold=-0.5)
        assert result is not None

    def test_plot_custom_title(self, minimal_brain_data):
        """Test custom title"""
        result = minimal_brain_data[0].plot(title="My Custom Title")
        assert result is not None

    def test_plot_matplotlib_axis(self, minimal_brain_data):
        """Test plotting on existing matplotlib axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = minimal_brain_data.plot(kind="timeseries", ax=ax)
        assert result is not None
        plt.close(fig)

        fig, ax = plt.subplots()
        result = minimal_brain_data[0].plot(kind="histogram", ax=ax)
        assert result is not None
        plt.close(fig)
