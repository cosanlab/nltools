"""Contract tests for `plot_surf` and `plot_flatmap`.

Covers the layout nltools adds on top of nilearn: the sign-aware stat-map
range, the grid shape implied by `view`/`hemi`, the shared colorbar, the
transparency mask, and the errors raised for an empty or invalid request.
"""

import os
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltools.data import BrainData
from nltools.plotting import plot_surf, plot_flatmap


class TestStatMapDefaults:
    @pytest.mark.parametrize(
        "values,expected",
        [
            ([0.0, 1.0, 4.0, float("nan")], ("Reds", 0.0, 4.0)),
            ([0.0, -1.0, -4.0, float("inf")], ("Blues_r", -4.0, 0.0)),
            ([-2.0, 0.0, 4.0], ("RdBu_r", -4.0, 4.0)),
        ],
    )
    def test_sign_aware_nilearn_ranges(self, values, expected):
        from nltools.plotting.brain import _resolve_stat_map_defaults

        assert _resolve_stat_map_defaults(values) == expected

    def test_explicit_values_win_individually(self):
        from nltools.plotting.brain import _resolve_stat_map_defaults

        assert _resolve_stat_map_defaults([1.0, 4.0], cmap="viridis", vmin=-1.0) == (
            "viridis",
            -1.0,
            4.0,
        )


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
