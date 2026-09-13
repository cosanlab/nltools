"""Tests for ``BrainData.iplot()`` — the self-owned niivue `anywidget` viewer.

Headless: we assert only Python-set widget *traits* — which volume byte
buffers are populated (``bg_bytes`` / ``statmap_bytes`` / ``atlas_bytes``), the
stat-map display params (``statmap`` dict + ``cal_min`` / ``cal_max``),
``slice_type``, ``colorbar``, ``atlas_outline``, ``atlas_lut``, and the
``slider_bounds``. Frontend-derived behavior (the WebGL render, hover labels,
4D scrubbing, right-drag windowing) lives in ``viewer.js`` and is exercised by
the browser smoke test, not here.

``iplot()`` returns a `_NiivueViewer` directly (no ipywidgets wrapper); the
in-widget threshold slider is native to the frontend, so ``controls`` only
toggles a trait.

All fast tests pass ``bg_img=False`` and use synthetic atlases: the
identity-affine ``minimal_brain_data`` fixture counts as standard space
(1mm), so auto-background would otherwise fetch a template from HuggingFace.
"""

import nibabel as nib
import numpy as np
import polars as pl
import pytest

from nltools.data.atlases import _Atlas
from nltools.data.braindata.viewer import _NiivueViewer


def _n_volumes(viewer):
    """Count populated volume buffers in the stack (bg / statmap / atlas)."""
    return sum(
        bool(b) for b in (viewer.bg_bytes, viewer.statmap_bytes, viewer.atlas_bytes)
    )


@pytest.fixture
def det_atlas():
    """Synthetic deterministic atlas with sparse indices (1, 2, 5)."""
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    arr.flat[:3] = [1, 2, 5]
    return _Atlas(
        name="synthdet",
        image=nib.Nifti1Image(arr, np.eye(4)),
        labels=pl.DataFrame({"index": [1, 2, 5], "name": ["a", "b", "c"]}),
        kind="deterministic",
        citation="synthetic",
    )


@pytest.fixture
def prob_atlas():
    """Synthetic probabilistic (4D) atlas."""
    return _Atlas(
        name="synthprob",
        image=nib.Nifti1Image(np.zeros((4, 4, 4, 2), np.float32), np.eye(4)),
        labels=pl.DataFrame({"index": [0, 1], "name": ["a", "b"]}),
        kind="probabilistic",
        citation="synthetic",
    )


class TestReturnType:
    def test_returns_niivue_viewer(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert isinstance(v, _NiivueViewer)

    def test_controls_default_on(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert v.controls is True

    def test_controls_false_toggles_trait(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, controls=False)
        assert v.controls is False

    def test_3d_has_only_statmap(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert _n_volumes(v) == 1
        assert v.statmap_bytes and not v.bg_bytes and not v.atlas_bytes

    def test_4d_loaded_as_single_volume(self, minimal_brain_data):
        # niivue scrubs 4D frames natively, so a stack is ONE volume buffer.
        v = minimal_brain_data.iplot(bg_img=False)
        assert _n_volumes(v) == 1
        assert minimal_brain_data.shape[0] == 50

    def test_default_slice_type_is_multiplanar(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert v.slice_type == "MULTIPLANAR"


class TestView:
    @pytest.mark.parametrize(
        "view,expected",
        [
            ("axial", "AXIAL"),
            ("coronal", "CORONAL"),
            ("sagittal", "SAGITTAL"),
            ("render", "RENDER"),
        ],
    )
    def test_view_sets_slice_type(self, minimal_brain_data, view, expected):
        v = minimal_brain_data[0].iplot(bg_img=False, view=view)
        assert v.slice_type == expected

    def test_invalid_view_raises(self, minimal_brain_data):
        with pytest.raises(ValueError, match="not recognized"):
            minimal_brain_data[0].iplot(bg_img=False, view="glass")

    def test_surface_raises_with_render_hint(self, minimal_brain_data):
        with pytest.raises(ValueError, match="render"):
            minimal_brain_data[0].iplot(bg_img=False, view="surface")

    def test_default_height_depends_on_view(self, minimal_brain_data):
        assert minimal_brain_data[0].iplot(bg_img=False).height == 600
        assert minimal_brain_data[0].iplot(bg_img=False, view="axial").height == 400


class TestThreshold:
    def test_threshold_sets_cal_min(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, threshold=2.3)
        assert v.cal_min == pytest.approx(2.3)
        # The unset ceiling is autoscaled (robust percentile), never None.
        assert v.cal_max is not None and v.cal_max > 0

    def test_lower_upper_set_window(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, lower=1.0, upper=2.0)
        assert v.cal_min == pytest.approx(1.0)
        assert v.cal_max == pytest.approx(2.0)

    def test_negative_lower_is_a_magnitude(self, minimal_brain_data):
        # The window is mirrored onto the negative limb, so a negative floor
        # means |v| >= 1 — never "everything above -1", which would include
        # every zero voxel outside the mask and paint the whole volume.
        v = minimal_brain_data[0].iplot(bg_img=False, lower=-1.0, upper=2.0)
        assert v.cal_min == pytest.approx(1.0)

    def test_lower_upper_take_precedence_over_threshold(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, threshold=2.3, upper=4.0)
        # lower/upper win: threshold is ignored; the floor is autoscaled
        # (epsilon above zero), not the requested 2.3.
        assert v.cal_min is not None and v.cal_min < 2.3
        assert v.cal_max == pytest.approx(4.0)

    def test_default_window_is_explicit(self, minimal_brain_data):
        """Python always owns the window: never None, so slider == render."""
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert v.cal_min is not None
        assert v.cal_max is not None


class TestColormap:
    def test_default_warm_with_winter_negative(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert v.statmap["colormap"] == "warm"
        assert v.statmap["colormap_negative"] == "winter"

    def test_viridis_passthrough(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, cmap="viridis")
        assert v.statmap["colormap"] == "viridis"

    def test_rdbu_r_maps_and_warns(self, minimal_brain_data):
        with pytest.warns(UserWarning, match="matplotlib"):
            v = minimal_brain_data[0].iplot(bg_img=False, cmap="RdBu_r")
        assert v.statmap["colormap"] == "warm"

    def test_default_palette_is_sign_aware(self, minimal_brain_data):
        positive = minimal_brain_data[0].copy()
        positive.data = np.abs(positive.data)
        negative = minimal_brain_data[0].copy()
        negative.data = -np.abs(negative.data)

        assert positive.iplot(bg_img=False).statmap["colormap"] == "warm"
        assert negative.iplot(bg_img=False).statmap["colormap_negative"] == "winter"


class TestColorbar:
    def test_colorbar_on_by_default(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert v.colorbar is True

    def test_colorbar_false_disables(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, colorbar=False)
        assert v.colorbar is False

    def test_explicit_is_colorbar_overrides_colorbar(self, minimal_brain_data):
        # An explicit is_colorbar kwarg wins over the colorbar= convenience.
        v = minimal_brain_data[0].iplot(bg_img=False, colorbar=True, is_colorbar=False)
        assert v.colorbar is False


class TestControls:
    def test_slider_bounds_are_a_positive_magnitude_window(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        data = minimal_brain_data[0].data
        assert v.slider_bounds["min"] > 0  # the floor can never reach zero
        assert v.slider_bounds["max"] == pytest.approx(float(np.nanmax(np.abs(data))))

    def test_zero_threshold_is_raised_above_zero(self, minimal_brain_data):
        # iplot(threshold=0) used to paint the whole volume box: every zero
        # voxel outside the mask passed the window.
        v = minimal_brain_data[0].iplot(bg_img=False, threshold=0.0)
        assert v.cal_min == pytest.approx(v.slider_bounds["min"])
        assert v.cal_min > 0

    def test_cal_window_is_reactive_trait(self, minimal_brain_data):
        # The frontend slider writes cal_min/cal_max; Python can set them too.
        v = minimal_brain_data[0].iplot(bg_img=False)
        v.cal_min, v.cal_max = -1.5, 3.0
        assert v.cal_min == pytest.approx(-1.5)
        assert v.cal_max == pytest.approx(3.0)

    def test_slider_initial_value_matches_threshold(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, lower=1.0, upper=2.0)
        assert v.slider_bounds["value_low"] == pytest.approx(1.0)
        assert v.slider_bounds["value_high"] == pytest.approx(2.0)


class TestKwargForwarding:
    def test_height_forwarded_to_trait(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, height=512)
        assert v.height == 512

    def test_extra_config_option_forwarded_to_niivue_opts(self, minimal_brain_data):
        # An unknown niivue ConfigOption rides through to niivue_opts verbatim.
        v = minimal_brain_data[0].iplot(bg_img=False, backColor=[0, 0, 0, 1])
        assert v.niivue_opts["backColor"] == [0, 0, 0, 1]


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
        v = minimal_brain_data[0].iplot(bg_img=False, atlas=det_atlas)
        assert _n_volumes(v) == 2  # statmap + atlas
        assert v.atlas_bytes
        # Dense LUT length == max_index + 1 == 6 (sparse indices 1, 2, 5).
        assert len(v.atlas_lut["labels"]) == 6

    def test_no_atlas_leaves_lut_empty(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False)
        assert v.atlas_lut == {}
        assert not v.atlas_bytes

    def test_outline_sets_atlas_outline(self, minimal_brain_data, det_atlas):
        v = minimal_brain_data[0].iplot(bg_img=False, atlas=det_atlas, outline=2.0)
        assert v.atlas_outline == pytest.approx(2.0)

    def test_filled_atlas_does_not_set_outline(self, minimal_brain_data, det_atlas):
        v = minimal_brain_data[0].iplot(bg_img=False, atlas=det_atlas, outline=0.0)
        assert v.atlas_outline == 0


@pytest.mark.slow
class TestRealAtlasAndBackground:
    """End-to-end paths that fetch real atlas / template files from HF."""

    def test_real_aal_overlay(self, minimal_brain_data):
        from nltools.data.atlases import load_atlas

        v = minimal_brain_data[0].iplot(bg_img=False, atlas="aal")
        assert _n_volumes(v) == 2
        expected = max(load_atlas("aal").labels["index"].to_list()) + 1
        assert len(v.atlas_lut["labels"]) == expected

    def test_auto_mni_background_loads_for_standard_space(self, minimal_brain_data):
        # Identity affine -> _is_standard_space True (1mm) -> fetch MNI bg.
        v = minimal_brain_data[0].iplot()  # bg_img default None == auto
        assert _n_volumes(v) == 2  # background + statmap
        assert v.bg_bytes


def _sparse_bd():
    """A (1, 27) map where most voxels are zero and one is an extreme outlier."""
    from nltools.data import BrainData

    affine = np.eye(4)
    mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int8), affine)
    data = np.zeros((1, 27), dtype=np.float32)
    data[0, :8] = [1.0, -2.0, 3.0, -4.0, 5.0, 2.5, -1.5, 40.0]  # 40 = outlier
    return BrainData(data, mask=mask)


class TestAutoscale:
    """#479: the default window comes from robust statistics, not extremes."""

    def test_default_window_is_robust(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False)
        vals = np.abs(bd.data[bd.data != 0])
        expected_hi = float(np.percentile(vals, 98))
        assert v.cal_max == pytest.approx(expected_hi)
        # Epsilon floor: tiny but positive, so zeros render transparent while
        # sub-threshold voxels stay visible.
        assert 0 < v.cal_min < 0.01 * v.cal_max

    def test_autoscale_false_uses_extremes_explicitly(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False, autoscale=False)
        # The raw range starts one slider step above zero, never at zero.
        assert v.cal_min == pytest.approx(v.slider_bounds["min"])
        assert v.cal_min > 0.0
        assert v.cal_max == pytest.approx(float(np.abs(bd.data).max()))

    def test_percentile_window_comes_from_lower_and_upper(self):
        """A custom percentile window is spelled with `lower`/`upper`.

        `autoscale` is a bool: robust default, or the raw extremes. There is
        deliberately no second spelling of a percentile window.
        """
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False, lower="60%", upper="98%")
        vals = np.abs(bd.data[bd.data != 0])
        assert v.cal_min == pytest.approx(float(np.percentile(vals, 60)))
        assert v.cal_max == pytest.approx(float(np.percentile(vals, 98)))

    def test_non_boolean_autoscale_raises(self):
        bd = _sparse_bd()
        with pytest.raises(TypeError, match="autoscale"):
            bd.iplot(bg_img=False, autoscale=(60, 98))

    def test_explicit_threshold_keeps_autoscaled_ceiling(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False, threshold=2.3)
        vals = np.abs(bd.data[bd.data != 0])
        assert v.cal_min == pytest.approx(2.3)
        assert v.cal_max == pytest.approx(float(np.percentile(vals, 98)))

    def test_percentile_string_thresholds(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False, upper="90%")
        vals = np.abs(bd.data[bd.data != 0])
        assert v.cal_max == pytest.approx(float(np.percentile(vals, 90)))

    def test_slider_handles_match_rendered_window(self):
        """The handles always show the window actually being rendered."""
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False)
        assert v.slider_bounds["value_low"] == pytest.approx(v.cal_min)
        assert v.slider_bounds["value_high"] == pytest.approx(v.cal_max)
        v2 = bd.iplot(bg_img=False, autoscale=False)
        assert v2.slider_bounds["value_low"] == pytest.approx(v2.cal_min)
        assert v2.slider_bounds["value_high"] == pytest.approx(v2.cal_max)

    def test_auto_symmetry_uses_symmetric_limbs_for_mixed_data(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False)
        assert v.cal_min_neg == pytest.approx(-v.cal_max)
        assert v.cal_max_neg == pytest.approx(-v.cal_min)
        assert v.mirror_negative is True

    def test_false_scales_positive_and_negative_limbs_independently(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False, symmetric=False)
        positives = bd.data[bd.data > 0]
        negatives = np.abs(bd.data[bd.data < 0])
        assert v.cal_max == pytest.approx(float(np.percentile(positives, 98)))
        assert v.cal_min_neg == pytest.approx(-float(np.percentile(negatives, 98)))
        assert v.mirror_negative is False

    def test_true_forces_symmetry_for_one_sided_data(self):
        bd = _sparse_bd()
        bd.data = np.abs(bd.data)
        v = bd.iplot(bg_img=False, symmetric=True)
        assert v.cal_min_neg == pytest.approx(-v.cal_max)
        assert v.cal_max_neg == pytest.approx(-v.cal_min)

    def test_invalid_symmetric_raises(self):
        with pytest.raises(TypeError, match="symmetric"):
            _sparse_bd().iplot(bg_img=False, symmetric="sometimes")
