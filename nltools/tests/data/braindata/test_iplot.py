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


class TestColormap:
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


class TestKwargForwarding:
    def test_height_forwarded_to_trait(self, minimal_brain_data):
        v = minimal_brain_data[0].iplot(bg_img=False, height=512)
        assert v.height == 512


class TestAtlas:
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

    def test_percentile_string_thresholds(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False, upper="90%")
        vals = np.abs(bd.data[bd.data != 0])
        assert v.cal_max == pytest.approx(float(np.percentile(vals, 90)))

    def test_auto_symmetry_uses_symmetric_limbs_for_mixed_data(self):
        bd = _sparse_bd()
        v = bd.iplot(bg_img=False)
        assert v.cal_min_neg == pytest.approx(-v.cal_max)
        assert v.cal_max_neg == pytest.approx(-v.cal_min)
        assert v.mirror_negative is True

    def test_invalid_symmetric_raises(self):
        with pytest.raises(TypeError, match="symmetric"):
            _sparse_bd().iplot(bg_img=False, symmetric="sometimes")
