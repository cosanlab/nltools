"""
Test suite for plot_surface() and plot_flatmap() functions.

Tests follow TDD approach: write tests first, then implement functionality.
Focuses on surface plotting functionality with intelligent hemisphere parsing.
"""

import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from nltools.data import BrainData
from nltools.plotting import plot_surface, plot_flatmap


def _surfaces_available():
    """Check if surface files are available for testing."""
    try:
        from nltools.plotting.brain import _get_surface_paths

        paths = _get_surface_paths()
        return os.path.exists(paths.get("pial_left", ""))
    except Exception:
        return False


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.skipif(
    not _surfaces_available(),
    reason="Surface files not available (nltools/resources/surfaces/)",
)
class TestSurfacePlot:
    """Test plot_surface() function"""

    def test_surface_resource_paths(self):
        """Test that surface files can be located and paths are valid"""
        from nltools.plotting.brain import _get_surface_paths
        import os

        paths = _get_surface_paths()

        # Should return dict with required surface types
        required_keys = [
            "pial_left",
            "pial_right",
            "inflated_left",
            "inflated_right",
            "curv_left",
            "curv_right",
        ]
        for key in required_keys:
            assert key in paths, f"Missing surface key: {key}"
            assert os.path.exists(paths[key]), f"Surface file not found: {paths[key]}"

    @pytest.mark.parametrize("input_type", ["brain_data", "nibabel", "file_path"])
    def test_surface_plot_input_types(self, sim_brain_data, tmpdir, input_type):
        """Test plot_surface accepts BrainData, nibabel image, or file path"""
        import nibabel as nib

        single_image = sim_brain_data[0]

        if input_type == "brain_data":
            fig = plot_surface(single_image)
        elif input_type == "nibabel":
            fig = plot_surface(single_image.to_nifti())
        elif input_type == "file_path":
            test_file = str(tmpdir / "test.nii.gz")
            nib.save(single_image.to_nifti(), test_file)
            fig = plot_surface(test_file)

        assert fig is not None
        plt.close(fig)

    def test_surface_plot_invalid_input(self):
        """Test error handling for invalid input types"""
        with pytest.raises((ValueError, TypeError)):
            plot_surface("nonexistent.nii.gz")
        with pytest.raises((ValueError, TypeError)):
            plot_surface([1, 2, 3])
        with pytest.raises((ValueError, TypeError)):
            plot_surface(None)

    def test_default_montage_layout(self, sim_brain_data):
        """Test default 2×2 montage structure"""
        single_image = sim_brain_data[0]
        fig = plot_surface(single_image)

        assert fig is not None
        assert hasattr(fig, "axes")
        assert len(fig.axes) == 4  # 2×2 grid
        plt.close(fig)

    def test_bilateral_surface_projection(self, sim_brain_data):
        """Test that volume projects correctly to both hemispheres"""
        from nilearn.surface import vol_to_surf
        from nltools.plotting.brain import _get_surface_paths

        single_image = sim_brain_data[0]
        nifti_img = single_image.to_nifti()
        paths = _get_surface_paths()

        # Project to both hemispheres
        left_texture = vol_to_surf(
            nifti_img,
            paths["pial_left"],
            interpolation="linear",
            n_samples=1,
            radius=0.0,
        )
        right_texture = vol_to_surf(
            nifti_img,
            paths["pial_right"],
            interpolation="linear",
            n_samples=1,
            radius=0.0,
        )

        assert left_texture.size > 0 and right_texture.size > 0
        assert len(left_texture.shape) == 1  # 1D vertex array

    def test_customization_options(self, sim_brain_data):
        """Test key customization parameters"""
        single_image = sim_brain_data[0]

        # Test various parameters
        fig = plot_surface(
            single_image,
            cmap="hot",
            threshold=0.5,
            vmax=1.0,
            vmin=-1.0,
            colorbar=False,
        )
        assert fig is not None
        plt.close(fig)

        # Test percentile threshold
        fig = plot_surface(single_image, threshold="95%")
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("bg_map", ["curvature", "sulc", None])
    def test_background_map_options(self, sim_brain_data, bg_map):
        """Test different background map options"""
        single_image = sim_brain_data[0]
        fig = plot_surface(single_image, bg_map=bg_map)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize(
        "hemi,view",
        [
            ("left", "lateral"),
            ("right", "lateral"),
            ("left", "medial"),
            ("right", "medial"),
        ],
    )
    def test_single_hemisphere_view(self, sim_brain_data, hemi, view):
        """Test plotting single hemisphere and view combinations"""
        single_image = sim_brain_data[0]
        fig = plot_surface(single_image, hemi=hemi, view=view)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("surface", ["pial", "inflated", "midthickness"])
    def test_surface_mesh_types(self, sim_brain_data, surface):
        """Test different surface mesh types"""
        single_image = sim_brain_data[0]
        fig = plot_surface(single_image, surface=surface, hemi="left", view="lateral")
        assert fig is not None
        plt.close(fig)

    def test_invalid_surface_type(self, sim_brain_data):
        """Test error handling for invalid surface type"""
        single_image = sim_brain_data[0]
        with pytest.raises(ValueError):
            plot_surface(single_image, surface="invalid")

    def test_empty_brain_data(self):
        """Test error handling for empty BrainData"""
        empty_brain = BrainData()
        with pytest.raises(ValueError, match="empty|Empty"):
            plot_surface(empty_brain)

    def test_multi_image_brain_data(self, sim_brain_data):
        """Test handling BrainData with multiple images"""
        # Should plot first image or mean
        fig = plot_surface(sim_brain_data)
        assert fig is not None
        plt.close(fig)

    def test_nan_inf_handling(self, sim_brain_data):
        """Test handling of NaN and Inf values"""
        single_image = sim_brain_data[0]
        single_image.data[0] = np.nan
        if len(single_image.data) > 1:
            single_image.data[1] = np.inf

        # Should handle gracefully with thresholding
        fig = plot_surface(single_image, threshold=0.5)
        assert fig is not None
        plt.close(fig)

    @pytest.mark.slow
    def test_save_functionality(self, sim_brain_data, tmpdir):
        """Test saving plot to file"""
        import os

        single_image = sim_brain_data[0]
        save_path = str(tmpdir / "plot_surface.png")

        fig = plot_surface(single_image, save=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_custom_axes_and_figsize(self, sim_brain_data):
        """Test custom axes and figure size"""
        single_image = sim_brain_data[0]

        # Custom figure size
        fig = plot_surface(single_image, figsize=(12, 12))
        assert fig.get_size_inches()[0] == 12
        plt.close(fig)

        # Custom axes
        fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
        result = plot_surface(single_image, axes=axes)
        assert result is not None
        plt.close(fig)

    def test_projection_parameters(self, sim_brain_data):
        """Test custom vol_to_surf parameters"""
        single_image = sim_brain_data[0]

        fig = plot_surface(
            single_image,
            n_samples=5,
            radius_mm=2.0,
            interpolation="linear",
            hemi="left",
            view="lateral",
        )
        assert fig is not None
        plt.close(fig)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
class TestPlotFlatmap:
    """Test plot_flatmap() function"""

    @pytest.mark.slow
    def test_basic_flatmap(self, sim_brain_data):
        """Test basic flatmap rendering"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image)
        assert fig is not None
        assert hasattr(fig, "axes")
        plt.close(fig)

    @pytest.mark.parametrize("input_type", ["brain_data", "nibabel", "file_path"])
    def test_flatmap_input_types(self, sim_brain_data, tmpdir, input_type):
        """Test plot_flatmap accepts BrainData, nibabel image, or file path"""
        import nibabel as nib

        single_image = sim_brain_data[0]

        if input_type == "brain_data":
            fig = plot_flatmap(single_image)
        elif input_type == "nibabel":
            fig = plot_flatmap(single_image.to_nifti())
        elif input_type == "file_path":
            test_file = str(tmpdir / "test.nii.gz")
            nib.save(single_image.to_nifti(), test_file)
            fig = plot_flatmap(test_file)

        assert fig is not None
        plt.close(fig)

    def test_flatmap_threshold_float(self, sim_brain_data):
        """Test float threshold parameter"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, threshold=0.5)
        assert fig is not None
        plt.close(fig)

    def test_flatmap_threshold_percentile(self, sim_brain_data):
        """Test percentile threshold parameter"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, threshold="95%")
        assert fig is not None
        plt.close(fig)

    def test_flatmap_colormap(self, sim_brain_data):
        """Test custom colormap"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, cmap="hot", threshold=0.5)
        assert fig is not None
        plt.close(fig)

    def test_flatmap_vmin_vmax(self, sim_brain_data):
        """Test custom vmin/vmax"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, vmin=-2.0, vmax=2.0)
        assert fig is not None
        plt.close(fig)

    def test_flatmap_without_curvature(self, sim_brain_data):
        """Test flatmap without curvature background"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, with_curvature=False)
        assert fig is not None
        plt.close(fig)

    def test_flatmap_curvature_parameters(self, sim_brain_data):
        """Test curvature contrast and brightness parameters"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(
            single_image,
            curvature_contrast=0.8,
            curvature_brightness=0.3,
        )
        assert fig is not None
        plt.close(fig)

    @pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
    def test_flatmap_colorbar_orientation(self, sim_brain_data, orientation):
        """Test colorbar orientation options"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(
            single_image,
            colorbar=True,
            colorbar_orientation=orientation,
        )
        assert fig is not None
        plt.close(fig)

    def test_flatmap_no_colorbar(self, sim_brain_data):
        """Test flatmap without colorbar"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, colorbar=False)
        assert fig is not None
        plt.close(fig)

    def test_flatmap_title(self, sim_brain_data):
        """Test custom title"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, title="Test Flatmap")
        assert fig is not None
        plt.close(fig)

    def test_flatmap_figsize(self, sim_brain_data):
        """Test custom figure size"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(single_image, figsize=(14, 8))
        assert fig.get_size_inches()[0] == pytest.approx(14, abs=0.5)
        plt.close(fig)

    def test_flatmap_custom_axes(self, sim_brain_data):
        """Test plotting on custom axes"""
        single_image = sim_brain_data[0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        result = plot_flatmap(single_image, axes=ax)
        assert result is not None
        plt.close(fig)

    @pytest.mark.slow
    def test_flatmap_save(self, sim_brain_data, tmpdir):
        """Test saving flatmap to file"""
        import os

        single_image = sim_brain_data[0]
        save_path = str(tmpdir / "flatmap.png")
        fig = plot_flatmap(single_image, save=save_path)
        assert os.path.exists(save_path)
        plt.close(fig)

    def test_flatmap_empty_brain_data(self):
        """Test error handling for empty BrainData"""
        empty_brain = BrainData()
        with pytest.raises(ValueError, match="empty|Empty"):
            plot_flatmap(empty_brain)

    def test_flatmap_multi_image_brain_data(self, sim_brain_data):
        """Test handling BrainData with multiple images"""
        # Should plot first image
        fig = plot_flatmap(sim_brain_data)
        assert fig is not None
        plt.close(fig)

    def test_flatmap_projection_parameters(self, sim_brain_data):
        """Test custom projection parameters"""
        single_image = sim_brain_data[0]
        fig = plot_flatmap(
            single_image,
            radius_mm=5.0,
            interpolation="nearest_most_frequent",
        )
        assert fig is not None
        plt.close(fig)


class TestBrainDataPlotFlatmap:
    """Test BrainData.plot_flatmap() method"""

    def test_brain_data_plot_flatmap(self, sim_brain_data):
        """Test BrainData.plot_flatmap() method"""
        single_image = sim_brain_data[0]
        fig = single_image.plot_flatmap()
        assert fig is not None
        plt.close(fig)

    def test_brain_data_plot_flatmap_with_threshold(self, sim_brain_data):
        """Test BrainData.plot_flatmap() with threshold"""
        single_image = sim_brain_data[0]
        fig = single_image.plot_flatmap(threshold=0.5, cmap="hot")
        assert fig is not None
        plt.close(fig)

    def test_brain_data_plot_flatmap_empty(self):
        """Test error handling for empty BrainData"""
        empty_brain = BrainData()
        with pytest.raises(ValueError, match="empty|Empty"):
            empty_brain.plot_flatmap()
