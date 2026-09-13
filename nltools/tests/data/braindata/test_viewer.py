"""Unit tests for the pure helpers in ``nltools.data.braindata.viewer``.

These cover palette generation, display-window and threshold-slider maths, and
background resolution, using synthetic arrays and affines — no network, no
browser, no widget construction. The LUT, colormap and slice-type helpers are
pinned through the facade in ``test_iplot.py``.
"""

import numpy as np
import pytest

from nltools.data.braindata.viewer import (
    _compute_display_window,
    _qualitative_colors,
    _resolve_background,
)


class _FakeBD:
    """Minimal stand-in exposing just the ``.data`` attribute the helper reads."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class TestQualitativeColors:
    def test_n_tuples_in_range(self):
        cs = _qualitative_colors(10)
        assert len(cs) == 10
        assert all(len(c) == 3 for c in cs)
        assert all(0 <= v <= 255 for c in cs for v in c)

    def test_deterministic(self):
        assert _qualitative_colors(5) == _qualitative_colors(5)
        assert _qualitative_colors(5, seed=1) != _qualitative_colors(5, seed=0)

    def test_distinct(self):
        assert len(set(_qualitative_colors(8))) == 8

    def test_zero(self):
        assert _qualitative_colors(0) == []

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _qualitative_colors(-1)


class TestComputeDisplayWindow:
    """Pin the percentile-string frame of reference for the display window.

    Percentile specs resolve against the finite nonzero magnitudes (C-10 pins
    this so the abs-array computation can be shared with `magnitudes` instead
    of being materialized three times per call).
    """

    def test_percentile_specs_resolve_against_finite_nonzero_magnitudes(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        data[::7] = 0.0  # masked-out voxels
        data[3] = np.nan
        data[5] = np.inf

        finite = data[np.isfinite(data)]
        magnitudes = np.abs(finite[finite != 0])

        window = _compute_display_window(data, threshold="60%")
        assert window.cal_min == pytest.approx(float(np.percentile(magnitudes, 60)))

        window = _compute_display_window(data, lower="10%", upper="90%")
        assert window.cal_min == pytest.approx(float(np.percentile(magnitudes, 10)))
        assert window.cal_max == pytest.approx(float(np.percentile(magnitudes, 90)))

    def test_numeric_specs_are_taken_as_magnitudes(self):
        """The window is mirrored onto the negative limb, so an edge's sign is moot.

        `lower=-1.0` means |v| >= 1 — never "everything above -1", which would
        admit every zero voxel outside the mask and paint the whole volume.
        """
        window = _compute_display_window(
            np.array([-3.0, 0.0, 1.0, 4.0]), lower=-1.0, upper=2.0
        )
        assert (window.cal_min, window.cal_max) == (1.0, 2.0)

    def test_default_floor_never_hides_a_nonzero_voxel(self):
        """The autoscale floor sits at or below the smallest nonzero magnitude.

        The floor exists to make stored zeros transparent, not to threshold.
        A fixed fraction of the ceiling breaks that promise once a map's
        dynamic range exceeds 1 / the fraction: the smallest real voxels fall
        below the floor and render transparent, which is data loss the user
        never asked for.
        """
        # P98 ~ 1.0, so a ceiling-fraction floor lands at 1e-6 -- three orders
        # of magnitude above the smallest real voxel.
        data = np.concatenate([np.full(500, 1.0), np.array([1e-9])])

        window = _compute_display_window(data)

        smallest_nonzero = np.abs(data[data != 0]).min()
        assert 0.0 < window.cal_min <= smallest_nonzero

    def test_autoscale_must_be_a_bool(self):
        """`autoscale` is bool-only; percentile windows use lower/upper."""
        data = np.array([-3.0, 0.0, 1.0, 4.0])
        for bad in [(60, 98), None, "98%", 98]:
            with pytest.raises(TypeError, match="autoscale"):
                _compute_display_window(data, autoscale=bad)

    def test_the_slider_bound_never_exceeds_the_resolved_floor(self):
        """``viewer.js`` clamps the applied floor to this bound.

        The frontend takes ``slider_bounds['min']`` as its floor epsilon, so a
        bound above the resolved ``cal_min`` would raise the rendered floor
        above the window Python resolved — #479's regression, from the browser.
        """
        rng = np.random.default_rng(0)
        data = rng.standard_normal(2000)
        data[::7] = 0.0

        window = _compute_display_window(data)

        assert 0.0 < window.slider_min <= window.cal_min

    def test_a_one_sided_limb_never_inverts_the_window(self):
        """A per-limb percentile ceiling must not land below an explicit floor."""
        data = np.concatenate([np.linspace(0.1, 7.0, 200), np.array([9.0])])

        window = _compute_display_window(data, threshold=8.5, symmetric=False)

        assert window.cal_max >= window.cal_min
        assert window.slider_value_high >= window.slider_value_low
        assert window.cal_min_neg <= window.cal_max_neg

    def test_default_floor_stays_above_zero_for_ordinary_maps(self):
        """The floor is still a positive epsilon so exact zeros stay transparent."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        data[::7] = 0.0

        window = _compute_display_window(data)

        assert window.cal_min > 0.0
        assert window.cal_min < window.cal_max


class TestBuildViewerRequiresWindow:
    """``_build_viewer`` cannot be constructed without an explicit window.

    An optional window is how the renderer and the slider handles drifted
    apart in the first place: niivue autoscaled from ``NaN`` while the
    handles showed the raw data extremes. Requiring the resolved window
    makes that divergence unrepresentable at construction.
    """

    def test_missing_window_raises(self):
        from nltools.data.braindata.viewer import _build_viewer

        with pytest.raises(TypeError, match="window"):
            _build_viewer(_FakeBD([-3.0, 0.0, 4.0]))


class TestDisplayWindowSliderBounds:
    """The slider fields of `_DisplayWindow` span the data and the window."""

    def test_bounds_are_a_magnitude_window_with_a_positive_floor(self):
        window = _compute_display_window([-3.0, 0.0, 4.0], lower=1.0, upper=4.0)
        assert window.slider_max == pytest.approx(4.0)  # max |v|
        assert window.slider_step == pytest.approx(4.0 / 200.0)
        assert window.slider_min > 0.0  # never zero
        assert window.slider_value_low == pytest.approx(1.0)
        assert window.slider_value_high == pytest.approx(4.0)

    def test_negative_dominated_data_sets_the_ceiling_by_magnitude(self):
        window = _compute_display_window([-9.0, 0.0, 4.0])
        assert window.slider_max == pytest.approx(9.0)
        assert window.slider_min > 0.0

    def test_a_zero_floor_is_raised_above_zero(self):
        """A floor of zero painted the whole volume box in the negative colormap."""
        window = _compute_display_window([-3.0, 4.0], lower=0.0)
        assert window.slider_value_low == pytest.approx(window.slider_min)
        assert window.slider_value_low > 0.0

    def test_ignores_nonfinite(self):
        window = _compute_display_window(
            [np.nan, -2.0, np.inf, 5.0], lower=0.0, upper=1.0
        )
        assert window.slider_min > 0.0
        assert window.slider_max == pytest.approx(5.0)

    def test_requested_window_sets_handles(self):
        window = _compute_display_window([-3.0, 4.0], lower=1.0, upper=3.0)
        assert window.slider_value_low == pytest.approx(1.0)
        assert window.slider_value_high == pytest.approx(3.0)

    def test_requested_window_widens_bounds(self):
        # A ceiling beyond the data range widens the bounds so the handles land
        # exactly where requested rather than being clamped to the extremes.
        window = _compute_display_window([-3.0, 4.0], lower=1.0, upper=99.0)
        assert window.slider_max == pytest.approx(99.0)
        assert window.slider_value_low == pytest.approx(1.0)
        assert window.slider_value_high == pytest.approx(99.0)
        assert 0.0 < window.slider_min < window.slider_value_low

    def test_empty_data_falls_back(self):
        window = _compute_display_window([])
        assert window.slider_max == pytest.approx(1.0)
        assert window.slider_min > 0.0
        assert window.slider_step > 0

    def test_all_zero_data_falls_back(self):
        window = _compute_display_window([0.0, 0.0, 0.0])
        assert window.slider_max == pytest.approx(1.0)
        assert window.slider_min > 0.0

    def test_constant_data(self):
        window = _compute_display_window([2.0, 2.0, 2.0], lower=2.0, upper=2.0)
        assert window.slider_max == pytest.approx(2.0)
        assert 0.0 < window.slider_min < window.slider_max


class TestResolveBackground:
    def test_false_disables_background(self):
        assert _resolve_background(np.eye(4), False) is None

    def test_string_path_passthrough(self):
        assert _resolve_background(np.eye(4), "/tmp/bg.nii.gz") == "/tmp/bg.nii.gz"

    def test_none_with_nonstandard_affine_is_none(self):
        # Non-isotropic affine is not standard space, so auto resolves to no
        # background without any network fetch.
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        assert _resolve_background(affine, None) is None


def test_gzip_nifti_is_deterministic_across_calls():
    """Same payload → identical bytes (gzip MTIME pinned), so content-addressed
    consumers de-duplicate identical volumes."""
    import gzip
    import time

    from nltools.data.braindata.viewer import _gzip_nifti

    raw = b"\x5c\x01\x00\x00" + bytes(range(256)) * 16
    a = _gzip_nifti(raw)
    time.sleep(1.1)
    b = _gzip_nifti(raw)
    assert a == b
    assert gzip.decompress(a) == raw
    assert _gzip_nifti(a) == a  # already-gzipped input passes through
