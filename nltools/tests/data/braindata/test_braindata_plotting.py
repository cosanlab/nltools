"""
Test suite for BrainData.plot() method.

Tests follow TDD approach: write tests first, then implement functionality.
Focuses on plotting functionality, brain-space integration, and user-friendly defaults.
"""

import pytest
import numpy as np
from nltools.data import BrainData


class TestBrainDataPlotting:
    """Test BrainData plotting methods."""

    # ==================== Phase 1: Baseline Tests ====================

    @pytest.mark.parametrize("method", ["glass", "slices"])
    def test_plot_non_finite_voxels_is_silent(self, minimal_brain_data, method):
        """NaN/inf voxels (ROI maps, tSNR with zero std) plot without nilearn's warning."""
        import warnings

        bd = minimal_brain_data[0].copy()
        bd.data[: bd.data.size // 3] = np.nan
        bd.data[-1] = np.inf
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Non-finite values detected")
            result = bd.plot(method=method)
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

    def test_plot_percentile_threshold_uses_nonzero_magnitudes(
        self, minimal_brain_data, monkeypatch
    ):
        """Percentile thresholds use the same magnitude frame as ``iplot``."""
        import matplotlib.pyplot as plt
        import nilearn.plotting

        bd = minimal_brain_data[0].copy()
        bd.data[:] = 0
        bd.data[:4] = [-4.0, -2.0, 1.0, 8.0]
        expected = float(np.percentile([4.0, 2.0, 1.0, 8.0], 75))
        captured = {}
        fig = plt.figure()

        class Display:
            frame_axes = type("FrameAxes", (), {"figure": fig})()

        def fake_plot_glass_brain(*args, **kwargs):
            captured.update(kwargs)
            return Display()

        monkeypatch.setattr(nilearn.plotting, "plot_glass_brain", fake_plot_glass_brain)
        bd.plot(method="glass", threshold="75%")

        assert captured["threshold"] == pytest.approx(expected)
        assert captured["cmap"] == "Reds"
        plt.close(fig)


class TestDefaultStatColormap:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (np.array([0.0, 1.0, 3.0, np.nan]), "Reds"),
            (np.array([0.0, -1.0, -3.0, -np.inf]), "Blues_r"),
            (np.array([-3.0, 0.0, 1.0]), "RdBu_r"),
            (np.array([0.0, np.nan]), "RdBu_r"),
        ],
    )
    def test_uses_sign_of_finite_nonzero_values(self, data, expected):
        from nltools.data.braindata.plotting import auto_select_colormap

        assert auto_select_colormap(data) == expected

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
