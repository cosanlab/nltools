"""Unit tests for the pure helpers in ``nltools.data.braindata.viewer``.

These cover palette generation, display-window and threshold-slider maths, and
background resolution, using synthetic arrays and affines — no network, no
browser, no widget construction. The LUT, colormap and slice-type helpers are
pinned through the facade in ``test_iplot.py``.
"""

import numpy as np
import pytest

from nltools.data.braindata.viewer import (
    qualitative_colors,
    resolve_background,
    threshold_slider_bounds,
)


class _FakeBD:
    """Minimal stand-in exposing just the ``.data`` attribute the helper reads."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class TestQualitativeColors:
    def test_n_tuples_in_range(self):
        cs = qualitative_colors(10)
        assert len(cs) == 10
        assert all(len(c) == 3 for c in cs)
        assert all(0 <= v <= 255 for c in cs for v in c)

    def test_deterministic(self):
        assert qualitative_colors(5) == qualitative_colors(5)
        assert qualitative_colors(5, seed=1) != qualitative_colors(5, seed=0)

    def test_distinct(self):
        assert len(set(qualitative_colors(8))) == 8

    def test_zero(self):
        assert qualitative_colors(0) == []

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            qualitative_colors(-1)


class TestComputeDisplayWindow:
    """Pin the percentile-string frame of reference for the display window.

    Percentile specs resolve against the finite nonzero magnitudes (C-10 pins
    this so the abs-array computation can be shared with `magnitudes` instead
    of being materialized three times per call).
    """

    def test_percentile_specs_resolve_against_finite_nonzero_magnitudes(self):
        from nltools.data.braindata.viewer import compute_display_window

        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        data[::7] = 0.0  # masked-out voxels
        data[3] = np.nan
        data[5] = np.inf

        finite = data[np.isfinite(data)]
        magnitudes = np.abs(finite[finite != 0])

        floor, ceiling = compute_display_window(data, threshold="60%")
        assert floor == pytest.approx(float(np.percentile(magnitudes, 60)))

        floor, ceiling = compute_display_window(data, lower="10%", upper="90%")
        assert floor == pytest.approx(float(np.percentile(magnitudes, 10)))
        assert ceiling == pytest.approx(float(np.percentile(magnitudes, 90)))

    def test_numeric_specs_pass_through_unchanged(self):
        from nltools.data.braindata.viewer import compute_display_window

        floor, ceiling = compute_display_window(
            np.array([-3.0, 0.0, 1.0, 4.0]), lower=-1.0, upper=2.0
        )
        assert (floor, ceiling) == (-1.0, 2.0)

    def test_default_floor_never_hides_a_nonzero_voxel(self):
        """The autoscale floor sits at or below the smallest nonzero magnitude.

        The floor exists to make stored zeros transparent, not to threshold.
        A fixed fraction of the ceiling breaks that promise once a map's
        dynamic range exceeds 1 / the fraction: the smallest real voxels fall
        below the floor and render transparent, which is data loss the user
        never asked for.
        """
        from nltools.data.braindata.viewer import compute_display_window

        # P98 ~ 1.0, so a ceiling-fraction floor lands at 1e-6 -- three orders
        # of magnitude above the smallest real voxel.
        data = np.concatenate([np.full(500, 1.0), np.array([1e-9])])

        floor, ceiling = compute_display_window(data)

        smallest_nonzero = np.abs(data[data != 0]).min()
        assert 0.0 < floor <= smallest_nonzero

    def test_autoscale_must_be_a_bool(self):
        """`autoscale` is bool-only; percentile windows use lower/upper."""
        from nltools.data.braindata.viewer import compute_display_window

        data = np.array([-3.0, 0.0, 1.0, 4.0])
        for bad in [(60, 98), None, "98%", 98]:
            with pytest.raises(TypeError, match="autoscale"):
                compute_display_window(data, autoscale=bad)

    def test_default_floor_stays_above_zero_for_ordinary_maps(self):
        """The floor is still a positive epsilon so exact zeros stay transparent."""
        from nltools.data.braindata.viewer import compute_display_window

        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        data[::7] = 0.0

        floor, ceiling = compute_display_window(data)

        assert floor > 0.0
        assert floor < ceiling


class TestBuildViewerRequiresWindow:
    """``build_viewer`` cannot be constructed without an explicit window.

    An optional window is how the renderer and the slider handles drifted
    apart in the first place: niivue autoscaled from ``NaN`` while the
    handles showed the raw data extremes. Making both edges required means
    that divergence is unrepresentable at construction.
    """

    def test_missing_both_edges_raises(self):
        from nltools.data.braindata.viewer import build_viewer

        with pytest.raises(TypeError, match="cal_min"):
            build_viewer(_FakeBD([-3.0, 0.0, 4.0]))

    def test_missing_one_edge_raises(self):
        from nltools.data.braindata.viewer import build_viewer

        with pytest.raises(TypeError, match="cal_max"):
            build_viewer(_FakeBD([-3.0, 0.0, 4.0]), cal_min=1.0)


class TestThresholdSliderBounds:
    def test_bounds_span_finite_data(self):
        lo, hi, vlo, vhi, step = threshold_slider_bounds(
            _FakeBD([-3.0, 0.0, 4.0]), cal_min=None, cal_max=None
        )
        assert lo == pytest.approx(-3.0)
        assert hi == pytest.approx(4.0)
        # No requested window -> handles sit at the data extremes.
        assert vlo == pytest.approx(-3.0)
        assert vhi == pytest.approx(4.0)
        assert step == pytest.approx(7.0 / 200.0)

    def test_ignores_nonfinite(self):
        lo, hi, *_ = threshold_slider_bounds(
            _FakeBD([np.nan, -2.0, np.inf, 5.0]), cal_min=None, cal_max=None
        )
        assert lo == pytest.approx(-2.0)
        assert hi == pytest.approx(5.0)

    def test_requested_window_sets_handles(self):
        _, _, vlo, vhi, _ = threshold_slider_bounds(
            _FakeBD([-3.0, 4.0]), cal_min=1.0, cal_max=3.0
        )
        assert vlo == pytest.approx(1.0)
        assert vhi == pytest.approx(3.0)

    def test_requested_window_widens_bounds(self):
        # A window outside the data range widens the bounds so the handles land
        # exactly where requested rather than being clamped to the extremes.
        lo, hi, vlo, vhi, _ = threshold_slider_bounds(
            _FakeBD([-3.0, 4.0]), cal_min=-99.0, cal_max=99.0
        )
        assert lo == pytest.approx(-99.0)
        assert hi == pytest.approx(99.0)
        assert vlo == pytest.approx(-99.0)
        assert vhi == pytest.approx(99.0)

    def test_empty_data_falls_back(self):
        lo, hi, vlo, vhi, step = threshold_slider_bounds(
            _FakeBD([]), cal_min=None, cal_max=None
        )
        assert (lo, hi) == (0.0, 1.0)
        assert step > 0

    def test_constant_data_widens_upper_bound(self):
        lo, hi, *_ = threshold_slider_bounds(
            _FakeBD([2.0, 2.0, 2.0]), cal_min=None, cal_max=None
        )
        assert lo == pytest.approx(2.0)
        assert hi == pytest.approx(3.0)  # lo + 1 so the range is non-degenerate


class TestResolveBackground:
    def test_false_disables_background(self):
        assert resolve_background(np.eye(4), False) is None

    def test_string_path_passthrough(self):
        assert resolve_background(np.eye(4), "/tmp/bg.nii.gz") == "/tmp/bg.nii.gz"

    def test_none_with_nonstandard_affine_is_none(self):
        # Non-isotropic affine is not standard space, so auto resolves to no
        # background without any network fetch.
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        assert resolve_background(affine, None) is None
