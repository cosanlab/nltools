"""Tests for ``BrainData.iplot()`` — the niivue (ipyniivue) interactive viewer.

Headless: we assert only Python-set widget state — volume count, per-volume
display props (cal_min/cal_max/colormap/colormap_negative/colormap_label),
``opts.slice_type``, settable ``frame_4d``, and the frame count we derive from
``bd.shape[0]``. Frontend-derived traits (``n_frame_4d``, crosshair/intensity/
hover/render) are never asserted.

All fast tests pass ``bg_img=False`` and use synthetic atlases: the
identity-affine ``minimal_brain_data`` fixture counts as standard space
(1mm), so auto-background would otherwise fetch a template from HuggingFace.
"""

import nibabel as nib
import numpy as np
import polars as pl
import pytest
from ipyniivue import NiiVue, SliceType

from nltools.data.atlases import Atlas


@pytest.fixture
def det_atlas():
    """Synthetic deterministic atlas with sparse indices (1, 2, 5)."""
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr.flat[:3] = [1, 2, 5]
    return Atlas(
        name="synthdet",
        image=nib.Nifti1Image(arr, np.eye(4)),
        labels=pl.DataFrame({"index": [1, 2, 5], "name": ["a", "b", "c"]}),
        kind="deterministic",
        citation="synthetic",
    )


@pytest.fixture
def prob_atlas():
    """Synthetic probabilistic (4D) atlas."""
    return Atlas(
        name="synthprob",
        image=nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), np.eye(4)),
        labels=pl.DataFrame({"index": [0, 1], "name": ["a", "b"]}),
        kind="probabilistic",
        citation="synthetic",
    )


class TestReturnType:
    def test_3d_returns_niivue_one_volume(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False)
        assert isinstance(nv, NiiVue)
        assert len(nv.volumes) == 1

    def test_4d_loaded_as_single_volume(self, minimal_brain_data):
        # niivue scrubs 4D frames natively, so a stack is ONE volume.
        nv = minimal_brain_data.iplot(bg_img=False)
        assert len(nv.volumes) == 1
        # The frame count we control is derived from the BrainData shape.
        assert minimal_brain_data.shape[0] == 50

    def test_default_slice_type_is_multiplanar(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False)
        assert nv.opts.slice_type == SliceType.MULTIPLANAR

    def test_frame_4d_is_settable(self, minimal_brain_data):
        nv = minimal_brain_data.iplot(bg_img=False)
        nv.volumes[0].frame_4d = 3
        assert nv.volumes[0].frame_4d == 3


class TestView:
    @pytest.mark.parametrize(
        "view,expected",
        [
            ("axial", SliceType.AXIAL),
            ("coronal", SliceType.CORONAL),
            ("sagittal", SliceType.SAGITTAL),
            ("render", SliceType.RENDER),
        ],
    )
    def test_view_sets_slice_type(self, minimal_brain_data, view, expected):
        nv = minimal_brain_data[0].iplot(bg_img=False, view=view)
        assert nv.opts.slice_type == expected

    def test_invalid_view_raises(self, minimal_brain_data):
        with pytest.raises(ValueError, match="not recognized"):
            minimal_brain_data[0].iplot(bg_img=False, view="glass")

    def test_surface_raises_with_render_hint(self, minimal_brain_data):
        with pytest.raises(ValueError, match="render"):
            minimal_brain_data[0].iplot(bg_img=False, view="surface")


class TestThreshold:
    def test_threshold_sets_cal_min(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False, threshold=2.3)
        assert nv.volumes[0].cal_min == pytest.approx(2.3)
        assert nv.volumes[0].cal_max is None

    def test_lower_upper_set_window(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False, lower=-1.0, upper=2.0)
        assert nv.volumes[0].cal_min == pytest.approx(-1.0)
        assert nv.volumes[0].cal_max == pytest.approx(2.0)

    def test_lower_upper_take_precedence_over_threshold(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False, threshold=2.3, upper=4.0)
        # lower/upper win: threshold is ignored, floor stays auto (None)
        assert nv.volumes[0].cal_min is None
        assert nv.volumes[0].cal_max == pytest.approx(4.0)

    def test_default_window_is_auto(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False)
        assert nv.volumes[0].cal_min is None
        assert nv.volumes[0].cal_max is None


class TestColormap:
    def test_default_warm_with_winter_negative(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False)
        assert nv.volumes[0].colormap == "warm"
        assert nv.volumes[0].colormap_negative == "winter"

    def test_viridis_passthrough(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False, cmap="viridis")
        assert nv.volumes[0].colormap == "viridis"

    def test_rdbu_r_maps_and_warns(self, minimal_brain_data):
        with pytest.warns(UserWarning, match="matplotlib"):
            nv = minimal_brain_data[0].iplot(bg_img=False, cmap="RdBu_r")
        assert nv.volumes[0].colormap == "warm"


class TestKwargForwarding:
    def test_height_forwarded_to_niivue(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False, height=512)
        assert nv.height == 512

    def test_config_option_forwarded_to_opts(self, minimal_brain_data):
        nv = minimal_brain_data[0].iplot(bg_img=False, is_colorbar=False)
        assert nv.opts.is_colorbar is False


class TestAtlas:
    def test_bad_atlas_type_raises(self, minimal_brain_data):
        with pytest.raises(TypeError, match="atlas must be"):
            minimal_brain_data[0].iplot(bg_img=False, atlas=123)

    def test_probabilistic_atlas_raises(self, minimal_brain_data, prob_atlas):
        with pytest.raises(ValueError, match="deterministic"):
            minimal_brain_data[0].iplot(bg_img=False, atlas=prob_atlas)

    def test_deterministic_atlas_adds_volume_with_lut(
        self, minimal_brain_data, det_atlas
    ):
        nv = minimal_brain_data[0].iplot(bg_img=False, atlas=det_atlas)
        assert len(nv.volumes) == 2  # statmap + atlas
        lut = nv.volumes[-1].colormap_label
        assert lut is not None
        # Dense LUT length == max_index + 1 == 6 (sparse indices 1, 2, 5).
        assert len(lut.labels) == 6

    def test_outline_sets_atlas_outline(self, minimal_brain_data, det_atlas):
        nv = minimal_brain_data[0].iplot(bg_img=False, atlas=det_atlas, outline=2.0)
        assert nv.opts.atlas_outline == pytest.approx(2.0)

    def test_filled_atlas_does_not_set_outline(self, minimal_brain_data, det_atlas):
        nv = minimal_brain_data[0].iplot(bg_img=False, atlas=det_atlas, outline=0.0)
        assert nv.opts.atlas_outline == 0


@pytest.mark.slow
class TestRealAtlasAndBackground:
    """End-to-end paths that fetch real atlas / template files from HF."""

    def test_real_aal_overlay(self, minimal_brain_data):
        from nltools.data.atlases import load_atlas

        nv = minimal_brain_data[0].iplot(bg_img=False, atlas="aal")
        assert len(nv.volumes) == 2
        expected = max(load_atlas("aal").labels["index"].to_list()) + 1
        assert len(nv.volumes[-1].colormap_label.labels) == expected

    def test_auto_mni_background_loads_for_standard_space(self, minimal_brain_data):
        # Identity affine -> is_standard_space True (1mm) -> fetch MNI bg.
        nv = minimal_brain_data[0].iplot()  # bg_img default None == auto
        assert len(nv.volumes) == 2  # background + statmap
