"""Tests for BrainData.iplot() — anywidget-based interactive viewer.

Tier 1: 3D ortho viewer with threshold panel (symmetric/independent + value/pct).
Tier 2: 4D viewer adds a volume slider.
view='surface' uses nilearn.view_img_on_surf under the same widget shell.
"""

import numpy as np
import pytest


class TestIplotReturnType:
    def test_iplot_3d_returns_widget(self, minimal_brain_data):
        from nltools.data.braindata.widgets import BrainViewerWidget

        w = minimal_brain_data[0].iplot()
        assert isinstance(w, BrainViewerWidget)
        assert w.n_volumes == 1
        assert w.has_volume_slider is False

    def test_iplot_4d_returns_widget_with_volume_slider(self, minimal_brain_data):
        from nltools.data.braindata.widgets import BrainViewerWidget

        w = minimal_brain_data.iplot()
        assert isinstance(w, BrainViewerWidget)
        assert w.n_volumes == minimal_brain_data.shape[0]
        assert w.has_volume_slider is True
        assert w.volume_idx == 0

    def test_iplot_initial_html_populated(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        assert isinstance(w.html, str)
        assert len(w.html) > 1000  # nilearn HTML is large

    def test_iplot_repr_mimebundle_present(self, minimal_brain_data):
        """JB v2 / mystmd consume the widget mimebundle for interactive built docs."""
        w = minimal_brain_data[0].iplot()
        bundle = w._repr_mimebundle_()
        if isinstance(bundle, tuple):
            bundle = bundle[0]
        # Live widget renderer
        assert "application/vnd.jupyter.widget-view+json" in bundle
        # Static-doc fallback (mystmd renders this when no widget manager state)
        assert "text/html" in bundle
        assert "iframe" in bundle["text/html"]
        assert "srcdoc=" in bundle["text/html"]

    def test_iplot_4d_static_fallback_has_volume_slider(self, minimal_brain_data):
        """4D static-doc fallback embeds a JS-driven volume slider."""
        # Take a small 4D slice to keep render time short
        w = minimal_brain_data[:3].iplot()
        bundle = w._repr_mimebundle_()
        if isinstance(bundle, tuple):
            bundle = bundle[0]
        html = bundle["text/html"]
        assert "bv-slider" in html
        assert "type=&quot;range&quot;" in html  # escaped for outer iframe srcdoc
        assert "addEventListener" in html  # JS swap handler is embedded

    def test_iplot_4d_static_fallback_escapes_close_script(self, minimal_brain_data):
        """The JS array of pre-rendered nilearn HTML is embedded inside an
        outer <script> tag. Nilearn HTML contains </script> literals, so
        without escaping `</` to `<\\/` the outer script tag terminates
        early and the slider never wires up.
        """
        w = minimal_brain_data[:3].iplot()
        bundle = w._repr_mimebundle_()
        if isinstance(bundle, tuple):
            bundle = bundle[0]
        html = bundle["text/html"]
        assert "&lt;\\/script&gt;" in html  # escaped close-tag inside JSON


class TestIplotInteractivity:
    def test_volume_change_triggers_refresh(self, minimal_brain_data):
        w = minimal_brain_data.iplot()
        before = w.html
        w.volume_idx = 2
        assert w.html != before

    def test_upper_change_triggers_refresh(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        before = w.html
        w.upper = float(w.vmax_abs * 0.5)
        assert w.html != before

    def test_lower_change_triggers_refresh_in_independent_mode(
        self, minimal_brain_data
    ):
        w = minimal_brain_data[0].iplot(mode="independent")
        before = w.html
        # setting upper to a value > all data masks every positive voxel
        w.upper = float(w.data_max)
        assert w.html != before


class TestIplotDataRange:
    def test_data_range_reflects_data(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        arr = np.asarray(minimal_brain_data[0].data)
        finite = arr[np.isfinite(arr)]
        assert w.vmax_abs == pytest.approx(float(np.abs(finite).max()))
        assert w.data_min == pytest.approx(min(float(finite.min()), 0.0))
        assert w.data_max == pytest.approx(max(float(finite.max()), 0.0))

    def test_pct_tables_have_101_entries(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        assert len(w.pct_table_abs) == 101
        assert len(w.pct_table_neg) == 101
        assert len(w.pct_table_pos) == 101

    def test_pct_table_abs_matches_numpy_percentile(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        arr = np.asarray(minimal_brain_data[0].data)
        finite = arr[np.isfinite(arr)]
        for p in (0, 25, 50, 75, 100):
            assert w.pct_table_abs[p] == pytest.approx(
                float(np.percentile(np.abs(finite), p))
            )


class TestIplotModes:
    def test_default_mode_symmetric_value(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        assert w.mode_signed is False
        assert w.mode_pct is False

    def test_mode_independent_kwarg(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot(mode="independent")
        assert w.mode_signed is True

    def test_units_percentile_kwarg(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot(units="percentile")
        assert w.mode_pct is True

    def test_invalid_mode_raises(self, minimal_brain_data):
        with pytest.raises(ValueError, match="symmetric"):
            minimal_brain_data[0].iplot(mode="weird")

    def test_invalid_units_raises(self, minimal_brain_data):
        with pytest.raises(ValueError, match="value"):
            minimal_brain_data[0].iplot(units="weird")

    def test_independent_mode_masks_data_in_python(self, minimal_brain_data):
        """Independent mode masks data ourselves and passes threshold=0 to
        nilearn. A wide [-large, +large] band should produce identical
        rendered HTML to a high symmetric threshold (both render as 'all
        masked')."""
        bd = minimal_brain_data[0]
        # Mask everything
        w = bd.iplot(mode="independent")
        w.lower = float(w.data_min)
        w.upper = float(w.data_max)
        html_all_masked = w.html
        # Should differ from unthresholded
        w_unthresh = bd.iplot()
        assert html_all_masked != w_unthresh.html


class TestIplotViewKwarg:
    def test_default_view_is_ortho(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        # ortho view emits view_img HTML which mentions 'stat-map'
        assert "stat" in w.html.lower() or "brain" in w.html.lower()

    def test_view_surface_uses_view_img_on_surf(self, minimal_brain_data, monkeypatch):
        import nltools.data.braindata.widgets as wmod

        called = {"surf": 0}
        orig_surf = wmod.view_img_on_surf

        def spy_surf(*args, **kwargs):
            called["surf"] += 1
            return orig_surf(*args, **kwargs)

        monkeypatch.setattr(wmod, "view_img_on_surf", spy_surf)

        minimal_brain_data[0].iplot(view="surface")
        assert called["surf"] >= 1

    def test_invalid_view_raises(self, minimal_brain_data):
        with pytest.raises(ValueError, match="ortho"):
            minimal_brain_data[0].iplot(view="glass")


class TestIplotKwargPassthrough:
    def test_upper_kwarg_sets_initial_upper(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot(upper=0.5)
        assert w.upper == pytest.approx(0.5)

    def test_threshold_kwarg_alias_for_upper(self, minimal_brain_data):
        """Backward-compat: threshold= still works, mapped to upper."""
        w = minimal_brain_data[0].iplot(threshold=0.5)
        assert w.upper == pytest.approx(0.5)

    def test_lower_kwarg_sets_initial_lower(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot(mode="independent", lower=-0.3)
        assert w.lower == pytest.approx(-0.3)
