"""
Test suite for BrainData.plot() method.

Tests follow TDD approach: write tests first, then implement functionality.
Focuses on plotting functionality, brain-space integration, and user-friendly defaults.
"""

import pytest
import numpy as np
from nltools.data import BrainData
from nltools.templates import get_brainspace, with_brainspace


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
        result = minimal_brain_data[0].plot(method="glass")
        assert result is not None

    @pytest.mark.slow
    def test_plot_multi_slice(self, minimal_brain_data):
        """Test multi-slice visualization"""
        result = minimal_brain_data[0].plot(method="slices")
        assert result is not None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"upper": 0.5},
            {"upper": "95%"},
            {"lower": -0.5},
            {"upper": "90%", "lower": "10%"},
        ],
        ids=["float", "percentile", "lower", "bandpass"],
    )
    def test_plot_thresholding(self, minimal_brain_data, kwargs):
        """Test thresholding functionality with various inputs."""
        result = minimal_brain_data[0].plot(**kwargs)
        assert result is not None

    def test_plot_custom_cut_coords(self, minimal_brain_data):
        """Test custom cut coordinates"""
        result = minimal_brain_data[0].plot(view="xyz", cut_coords=[[0], [0], [0]])
        assert result is not None

    def test_plot_custom_colormap(self, minimal_brain_data):
        """Test custom colormap"""
        result = minimal_brain_data[0].plot(cmap="hot")
        assert result is not None

    def test_plot_respects_template_changes(self, minimal_brain_data):
        """Test that plot respects brain-space changes."""
        single_image = minimal_brain_data[0]
        with with_brainspace(template="nilearn", resolution=2):
            result = single_image.plot()
            assert result is not None

    def test_plot_empty_brain_data(self):
        """Test error handling for empty BrainData"""
        brain = BrainData()
        with pytest.raises(ValueError, match="empty|Empty"):
            brain.plot()

    @pytest.mark.parametrize("kind", ["invalid", "", 123, None])
    def test_plot_invalid_kind(self, minimal_brain_data, kind):
        """Test error handling for invalid 'kind' parameter"""
        with pytest.raises(ValueError):
            minimal_brain_data[0].plot(method=kind)

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
        result = single_image.plot(upper=0.5)
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
        result = single_image.plot(view="xyz", cut_coords=[[0], [0], [0]])
        assert result is not None
        result = single_image.plot(
            view="xyz",
            cut_coords=[range(-10, 11, 5), range(-10, 11, 5), range(-10, 11, 5)],
        )
        assert result is not None

    def test_plot_custom_bg_img_nibabel(self, minimal_brain_data):
        """Test custom background image as nibabel image"""
        import nibabel as nib

        single_image = minimal_brain_data[0]
        custom_bg = nib.load(get_brainspace().brain)
        result = single_image.plot(bg_img=custom_bg)
        assert result is not None

    def test_plot_save_functionality(self, minimal_brain_data, tmpdir):
        """Test save functionality"""
        single_image = minimal_brain_data[0]
        save_path = str(tmpdir / "test_plot.png")
        result = single_image.plot(method="glass", save=save_path)
        assert result is not None
        import os

        glass_file = save_path.replace(".png", "_glass.png")
        assert os.path.exists(glass_file)

    # ==================== Phase 4/5/6: User-Friendly Features ====================

    @pytest.mark.parametrize("stat", ["mean", "median", "std"])
    def test_plot_timeseries(self, minimal_brain_data, stat):
        """Test timeseries plotting with various aggregation stats."""
        result = minimal_brain_data.plot(method="timeseries", stat=stat)
        assert result is not None

    def test_plot_histogram(self, minimal_brain_data):
        """Test histogram plotting for single and multiple images."""
        result = minimal_brain_data[0].plot(method="histogram")
        assert result is not None
        result = minimal_brain_data.plot(method="histogram")
        assert result is not None

    def test_plot_threshold_convenience(self, minimal_brain_data):
        """`threshold` is an absolute-value transparency cutoff (nilearn semantics)."""
        result = minimal_brain_data[0].plot(threshold=0.5)
        assert result is not None
        result = minimal_brain_data[0].plot(threshold=0)
        assert result is not None
        with pytest.raises(ValueError, match="absolute-value"):
            minimal_brain_data[0].plot(threshold=-0.5)

    def test_plot_custom_title(self, minimal_brain_data):
        """Test custom title"""
        result = minimal_brain_data[0].plot(title="My Custom Title")
        assert result is not None

    def test_plot_matplotlib_axis(self, minimal_brain_data):
        """Test plotting on existing matplotlib axes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = minimal_brain_data.plot(method="timeseries", ax=ax)
        assert result is not None
        plt.close(fig)

        fig, ax = plt.subplots()
        result = minimal_brain_data[0].plot(method="histogram", ax=ax)
        assert result is not None
        plt.close(fig)

    # ==================== Multi-image rendering (limit) ====================

    def test_plot_multi_image_returns_list_glass(self, minimal_brain_data):
        """Multi-image glass plot returns a list of figures, one per image."""
        from matplotlib.figure import Figure

        with pytest.warns(UserWarning, match="plotting first"):
            result = minimal_brain_data.plot(method="glass", limit=2)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(f, Figure) for f in result)

    def test_plot_multi_image_returns_list_slices(self, minimal_brain_data):
        """Multi-image slices plot returns one figure per image-and-view pair."""
        from matplotlib.figure import Figure

        # `minimal_brain_data` has tiny bounds; supply in-range cut_coords.
        # `limit` caps the number of images rendered; slices still produces
        # one figure per axis, so the list has limit*len(view) figures.
        with pytest.warns(UserWarning, match="plotting first"):
            result = minimal_brain_data.plot(
                method="slices", view="xz", cut_coords=[[0], [0]], limit=2
            )
        assert isinstance(result, list)
        assert len(result) == 4  # 2 images x 2 views
        assert all(isinstance(f, Figure) for f in result)

    def test_plot_multi_image_no_warning_within_limit(self, minimal_brain_data):
        """No warning when image count is within `limit`."""
        import warnings

        sub = minimal_brain_data[:2]
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = sub.plot(method="glass", limit=3)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_plot_single_image_still_returns_figure(self, minimal_brain_data):
        """Single-image data continues to return a single Figure (not a list)."""
        from matplotlib.figure import Figure

        result = minimal_brain_data[0].plot(method="glass")
        assert isinstance(result, Figure)


# ---------------------------------------------------------------------------
# Standard-space gate (glass / slices / flatmap / surf on native data)
# ---------------------------------------------------------------------------


@pytest.fixture
def native_brain_data():
    """BrainData on a Miyawaki-shaped (anisotropic, native) affine.

    Used to exercise the standard-space plotting gate. Voxels (3.3, 3.6,
    6.4) mm fail :func:`nltools.templates.is_standard_space`.
    """
    import nibabel as nib

    np.random.seed(0)
    spatial_shape = (4, 4, 3)
    n_samples = 6
    affine = np.diag([3.3, 3.6, 6.4, 1.0]).astype(float)

    # Real binary mask (some zero voxels) so nilearn's transparency_range
    # has a usable [0, 1] span.
    mask_data = np.zeros(spatial_shape, dtype=np.float32)
    mask_data[:, :, :2] = 1.0
    mask_img = nib.Nifti1Image(mask_data, affine)
    volume_4d = np.random.randn(*spatial_shape, n_samples).astype(np.float32)
    nifti_img = nib.Nifti1Image(volume_4d, affine)
    return BrainData(nifti_img, mask=mask_img)


class TestStandardSpaceGate:
    def test_glass_on_native_raises(self, native_brain_data):
        with pytest.raises(ValueError, match="standard MNI space"):
            native_brain_data[0].plot(method="glass")

    def test_glass_on_native_falls_back_when_bg_img_provided(self, native_brain_data):
        """If the user passed a bg_img, redirect glass→slices with a warning."""
        # Use the BrainData's own mask as a stand-in bg_img — it shares the
        # native affine so plot_stat_map renders without resampling.
        with pytest.warns(UserWarning, match="glass.*falling back"):
            result = native_brain_data[0].plot(
                method="glass",
                bg_img=native_brain_data.mask,
                cut_coords=[[5]],
            )
        assert result is not None

    def test_slices_no_bg_on_native_raises(self, native_brain_data):
        with pytest.raises(ValueError, match="non-standard"):
            native_brain_data[0].plot(method="slices", cut_coords=[0])

    def test_slices_with_bg_on_native_works(self, native_brain_data):
        """Explicit bg_img is the supported escape hatch for native data."""
        result = native_brain_data[0].plot(
            method="slices",
            bg_img=native_brain_data.mask,
            cut_coords=[[5]],
        )
        assert result is not None

    def test_flatmap_on_native_raises(self, native_brain_data):
        with pytest.raises(ValueError, match="standard MNI space"):
            native_brain_data[0].plot_flatmap()

    def test_surf_on_native_raises(self, native_brain_data):
        with pytest.raises(ValueError, match="standard MNI space"):
            native_brain_data[0].plot_surf()
