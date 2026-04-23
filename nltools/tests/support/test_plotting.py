"""
Test suite for plot_surface() and plot_flatmap() functions.

Tests follow TDD approach: write tests first, then implement functionality.
Focuses on surface plotting functionality with intelligent hemisphere parsing.
"""

import os
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltools.data import BrainData
from nltools.plotting import plot_surf, plot_flatmap


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
class TestPlotSurf:
    """Test plot_surf() function — 2×2 surface montage with tight framing."""

    # ---- fast validation tests (no rendering / network) ----

    def test_plot_surf_empty_brain_raises(self):
        """Empty BrainData should raise ValueError before any surface work."""
        with pytest.raises(ValueError, match="empty|Empty"):
            plot_surf(BrainData())

    def test_plot_surf_invalid_hemi_raises(self, minimal_brain_data):
        """Unknown hemi string should raise ValueError up front."""
        with pytest.raises(ValueError, match="hemi"):
            plot_surf(minimal_brain_data[0], hemi="middle")

    def test_plot_surf_invalid_view_raises(self, minimal_brain_data):
        """Unknown view should raise ValueError up front."""
        with pytest.raises(ValueError, match="view"):
            plot_surf(minimal_brain_data[0], view="above")

    def test_plot_surf_is_exposed_from_plotting(self):
        """plot_surf must be importable from nltools.plotting."""
        from nltools.plotting import plot_surf as _ps

        assert callable(_ps)

    def test_plot_surf_is_method_on_braindata(self, minimal_brain_data):
        """BrainData should have a callable plot_surf method."""
        assert callable(getattr(minimal_brain_data, "plot_surf", None))

    # ---- rendering tests (slow; use fsaverage) ----

    @pytest.mark.slow
    def test_plot_surf_default_is_2x2_grid(self, sim_brain_data):
        """Default produces a 2×2 montage (2 views × 2 hemis)."""
        fig = plot_surf(sim_brain_data[0])
        axes_3d = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
        assert len(axes_3d) == 4
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_view_subsets_grid(self, sim_brain_data):
        """view=["lateral"] → 1 row × 2 hemi cols."""
        fig = plot_surf(sim_brain_data[0], view=["lateral"])
        axes_3d = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
        assert len(axes_3d) == 2
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_hemi_subsets_grid(self, sim_brain_data):
        """hemi='left' → 2 view rows × 1 hemi col."""
        fig = plot_surf(sim_brain_data[0], hemi="left")
        axes_3d = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
        assert len(axes_3d) == 2
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_single_view_single_hemi(self, sim_brain_data):
        """view='lateral', hemi='left' → single axis."""
        fig = plot_surf(sim_brain_data[0], view="lateral", hemi="left")
        axes_3d = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
        assert len(axes_3d) == 1
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_applies_zoom(self, sim_brain_data, monkeypatch):
        """zoom= kwarg is forwarded to Axes3D.set_box_aspect on every subplot."""
        from mpl_toolkits.mplot3d import Axes3D

        calls = []
        orig = Axes3D.set_box_aspect

        def spy(self, aspect, *, zoom=1):
            calls.append(zoom)
            return orig(self, aspect, zoom=zoom)

        monkeypatch.setattr(Axes3D, "set_box_aspect", spy)
        fig = plot_surf(sim_brain_data[0], zoom=1.4)
        # nilearn also calls set_box_aspect internally; we only care that
        # plot_surf applies zoom=1.4 once per subplot (4 subplots).
        assert calls.count(1.4) == 4
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_turns_axes_off(self, sim_brain_data):
        """Every 3D axis has its frame/grid hidden."""
        fig = plot_surf(sim_brain_data[0])
        axes_3d = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
        assert axes_3d  # sanity
        for ax in axes_3d:
            assert ax.axison is False
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_colorbar_is_shared(self, sim_brain_data):
        """colorbar=True produces exactly one shared colorbar, not one-per-subplot."""
        fig = plot_surf(sim_brain_data[0], colorbar=True)
        cbar_axes = [ax for ax in fig.axes if not isinstance(ax, Axes3D)]
        assert len(cbar_axes) == 1
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_colorbar_false(self, sim_brain_data):
        """colorbar=False produces no colorbar axis."""
        fig = plot_surf(sim_brain_data[0], colorbar=False)
        cbar_axes = [ax for ax in fig.axes if not isinstance(ax, Axes3D)]
        assert len(cbar_axes) == 0
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_transparency_auto_uses_mask(self, sim_brain_data, monkeypatch):
        """transparency='auto' resolves to the BrainData mask."""
        from nltools.plotting import brain as bmod

        captured = {}
        orig = bmod._resolve_transparency

        def spy(t, b):
            captured["t"] = t
            captured["b"] = b
            return orig(t, b)

        monkeypatch.setattr(bmod, "_resolve_transparency", spy)
        fig = plot_surf(sim_brain_data[0])
        assert captured["t"] == "auto"
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_braindata_method_returns_figure(self, sim_brain_data):
        """BrainData.plot_surf() facade returns a Figure with the 2×2 grid."""
        fig = sim_brain_data[0].plot_surf()
        axes_3d = [ax for ax in fig.axes if isinstance(ax, Axes3D)]
        assert len(axes_3d) == 4
        plt.close(fig)

    @pytest.mark.slow
    def test_plot_surf_save(self, sim_brain_data, tmpdir):
        """save= writes a file to disk."""
        save_path = str(tmpdir / "plot_surf.png")
        fig = plot_surf(sim_brain_data[0], save=save_path)
        assert os.path.exists(save_path)
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
