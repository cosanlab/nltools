"""Unit tests for the pure helpers in ``nltools.data.braindata.viewer``.

These cover the functional core (atlas LUT building, palette generation,
colormap / view / background resolution) with synthetic in-memory atlases
and affines — no network, no browser, no widget construction.
"""

import nibabel as nib
import numpy as np
import polars as pl
import pytest

from nltools.data.atlases import Atlas
from nltools.data.braindata.viewer import (
    atlas_to_label_lut,
    divergent_partner,
    qualitative_colors,
    resolve_background,
    resolve_cmap,
    slice_type_for,
    threshold_slider_bounds,
)


class _FakeBD:
    """Minimal stand-in exposing just the ``.data`` attribute the helper reads."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


def _atlas(indices, names, *, kind="deterministic"):
    """Build a synthetic in-memory Atlas from index/name pairs."""
    arr = np.zeros((4, 4, 4), dtype=np.int16)
    for k, idx in enumerate(indices):
        arr.flat[k] = idx
    img = nib.Nifti1Image(arr, np.eye(4))
    return Atlas(
        name="synth",
        image=img,
        labels=pl.DataFrame({"index": list(indices), "name": list(names)}),
        kind=kind,
        citation="synthetic",
    )


class TestAtlasToLabelLut:
    def test_lengths_dense_to_max_index(self):
        lut = atlas_to_label_lut(_atlas([1, 2, 3], ["a", "b", "c"]))
        for key in ("R", "G", "B", "A", "labels"):
            assert len(lut[key]) == 4  # max_index (3) + 1

    def test_index_zero_transparent(self):
        lut = atlas_to_label_lut(_atlas([1, 2, 3], ["a", "b", "c"]))
        assert lut["A"][0] == 0
        assert lut["labels"][0] == ""

    def test_names_and_colors_placed_at_index(self):
        lut = atlas_to_label_lut(_atlas([1, 2, 3], ["a", "b", "c"]))
        assert lut["labels"][1] == "a"
        assert lut["labels"][3] == "c"
        assert all(lut["A"][i] == 255 for i in (1, 2, 3))

    def test_sparse_index_gaps_transparent(self):
        lut = atlas_to_label_lut(_atlas([1, 2, 5], ["a", "b", "c"]))
        assert len(lut["labels"]) == 6  # dense to index 5
        assert lut["A"][3] == 0 and lut["A"][4] == 0
        assert lut["labels"][3] == "" and lut["labels"][4] == ""
        assert lut["A"][5] == 255 and lut["labels"][5] == "c"

    def test_deterministic(self):
        a = atlas_to_label_lut(_atlas([1, 2, 5], ["a", "b", "c"]))
        b = atlas_to_label_lut(_atlas([1, 2, 5], ["a", "b", "c"]))
        assert a == b

    def test_probabilistic_raises(self):
        with pytest.raises(ValueError, match="deterministic"):
            atlas_to_label_lut(_atlas([0, 1], ["a", "b"], kind="probabilistic"))


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


class TestResolveCmap:
    def test_niivue_name_passthrough(self):
        assert resolve_cmap("warm") == "warm"
        assert resolve_cmap("viridis") == "viridis"

    def test_case_insensitive(self):
        assert resolve_cmap("Warm") == "warm"

    def test_matplotlib_name_maps_and_warns(self):
        with pytest.warns(UserWarning, match="matplotlib"):
            assert resolve_cmap("RdBu_r") == "warm"

    def test_unknown_falls_back_and_warns(self):
        with pytest.warns(UserWarning, match="not a known"):
            assert resolve_cmap("not_a_colormap_xyz") == "warm"


class TestDivergentPartner:
    def test_warm_maps_to_winter(self):
        assert divergent_partner("warm") == "winter"

    def test_default_is_winter(self):
        assert divergent_partner("some_sequential") == "winter"


class TestSliceTypeFor:
    @pytest.mark.parametrize(
        "view,expected",
        [
            ("ortho", "MULTIPLANAR"),
            ("axial", "AXIAL"),
            ("coronal", "CORONAL"),
            ("sagittal", "SAGITTAL"),
            ("render", "RENDER"),
        ],
    )
    def test_valid_views(self, view, expected):
        assert slice_type_for(view) == expected

    def test_invalid_view_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            slice_type_for("glass")

    def test_surface_raises_with_render_hint(self):
        with pytest.raises(ValueError, match="render"):
            slice_type_for("surface")


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
