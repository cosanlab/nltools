"""Tests for BrainData.iplot() — anywidget-based interactive viewer.

Tier 1: 3D ortho viewer with threshold slider.
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
        assert 'type=&quot;range&quot;' in html  # escaped for outer iframe srcdoc
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
        # No raw "</script" should remain in the JSON array section
        # (only in the literal `</script>` that closes our IIFE)
        # Heuristic: count "</script" in the escaped inner_doc portion.
        # Easier: the escaped inner_doc has &lt;\/script&gt; for nilearn's
        # closing tags after the fix.
        assert "&lt;\\/script&gt;" in html  # escaped close-tag inside JSON


class TestIplotInteractivity:
    def test_volume_change_triggers_refresh(self, minimal_brain_data):
        w = minimal_brain_data.iplot()
        before = w.html
        w.volume_idx = 2
        assert w.html != before

    def test_threshold_change_triggers_refresh(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        before = w.html
        # pick a value within range
        w.threshold = float(w.threshold_max * 0.5)
        assert w.html != before

    def test_threshold_range_reflects_data(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot()
        expected_max = float(np.nanmax(np.abs(minimal_brain_data[0].data)))
        assert w.threshold_min == 0.0
        assert w.threshold_max == pytest.approx(expected_max)
        assert w.threshold_step == pytest.approx(expected_max / 100.0)


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
    def test_initial_threshold_kwarg(self, minimal_brain_data):
        w = minimal_brain_data[0].iplot(threshold=0.5)
        assert w.threshold == pytest.approx(0.5)
