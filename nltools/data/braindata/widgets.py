"""Anywidget-based interactive viewer for BrainData.

`BrainViewerWidget` wraps nilearn's HTML viewers (`view_img`,
`view_img_on_surf`) inside an iframe with a threshold slider, plus a
volume slider for 4D BrainData. Renders inline in Jupyter, marimo, and
Jupyter Book v2 (mystmd) static-built sites via the standard widget
mimebundle.
"""

from __future__ import annotations

import warnings

import numpy as np
import anywidget
import traitlets

from nilearn.plotting import view_img, view_img_on_surf


_ALLOWED_VIEWS = ("ortho", "surface")


def _html_escape_for_attr(s: str) -> str:
    """Escape an HTML doc for embedding in an iframe srcdoc attribute."""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class BrainViewerWidget(anywidget.AnyWidget):
    """Interactive ortho/surface viewer with threshold and (for 4D) volume sliders."""

    volume_idx = traitlets.Int(0).tag(sync=True)
    threshold = traitlets.Float(0.0).tag(sync=True)
    n_volumes = traitlets.Int(1).tag(sync=True)
    has_volume_slider = traitlets.Bool(False).tag(sync=True)
    threshold_min = traitlets.Float(0.0).tag(sync=True)
    threshold_max = traitlets.Float(1.0).tag(sync=True)
    threshold_step = traitlets.Float(0.01).tag(sync=True)
    html = traitlets.Unicode("").tag(sync=True)

    _esm = """
    function debounce(fn, ms) {
      let t;
      return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
    }
    function render({ model, el }) {
      el.innerHTML = `
        <div class="bv-controls">
          <div class="bv-row bv-volume-row" style="display:none">
            <label>Volume</label>
            <input type="range" class="bv-volume-slider" min="0" step="1"/>
            <span class="bv-volume-readout"></span>
          </div>
          <div class="bv-row">
            <label>Threshold</label>
            <input type="range" class="bv-threshold-slider"/>
            <span class="bv-threshold-readout"></span>
          </div>
        </div>
        <iframe class="bv-frame" sandbox="allow-scripts allow-same-origin"></iframe>
      `;
      const volRow = el.querySelector(".bv-volume-row");
      const volSlider = el.querySelector(".bv-volume-slider");
      const volReadout = el.querySelector(".bv-volume-readout");
      const thrSlider = el.querySelector(".bv-threshold-slider");
      const thrReadout = el.querySelector(".bv-threshold-readout");
      const frame = el.querySelector(".bv-frame");

      const syncControls = () => {
        const showVol = model.get("has_volume_slider");
        volRow.style.display = showVol ? "" : "none";
        if (showVol) {
          volSlider.max = String(model.get("n_volumes") - 1);
          volSlider.value = String(model.get("volume_idx"));
          volReadout.textContent = `${model.get("volume_idx")} / ${model.get("n_volumes") - 1}`;
        }
        thrSlider.min = String(model.get("threshold_min"));
        thrSlider.max = String(model.get("threshold_max"));
        thrSlider.step = String(model.get("threshold_step"));
        thrSlider.value = String(model.get("threshold"));
        thrReadout.textContent = Number(model.get("threshold")).toFixed(3);
      };

      const syncFrame = () => { frame.srcdoc = model.get("html") || ""; };

      const pushVolume = debounce((v) => model.set("volume_idx", v), 100);
      const pushThreshold = debounce((v) => model.set("threshold", v), 100);

      volSlider.addEventListener("input", (e) => {
        const v = Number(e.target.value);
        volReadout.textContent = `${v} / ${model.get("n_volumes") - 1}`;
        pushVolume(v); model.save_changes();
      });
      thrSlider.addEventListener("input", (e) => {
        const v = Number(e.target.value);
        thrReadout.textContent = v.toFixed(3);
        pushThreshold(v); model.save_changes();
      });

      model.on("change:html", syncFrame);
      model.on("change:volume_idx change:threshold change:n_volumes change:has_volume_slider change:threshold_min change:threshold_max change:threshold_step", syncControls);

      syncControls();
      syncFrame();
    }
    export default { render };
    """

    _css = """
    .bv-controls { font-family: sans-serif; padding: 4px 0; }
    .bv-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
    .bv-row label { min-width: 80px; font-size: 0.9em; }
    .bv-row input[type=range] { flex: 1; }
    .bv-row span { min-width: 80px; font-size: 0.85em; color: #555; }
    .bv-frame { width: 100%; min-height: 520px; border: 0; }
    """

    def __init__(
        self, bd, *, view: str = "ortho", threshold: float | None = None, **view_kwargs
    ):
        if view not in _ALLOWED_VIEWS:
            raise ValueError(f"view={view!r} not in {set(_ALLOWED_VIEWS)}.")
        super().__init__()
        self._bd = bd
        self._view = view
        self._kwargs = view_kwargs

        n = bd.shape[0] if len(bd.shape) > 1 else 1
        self.n_volumes = n
        self.has_volume_slider = n > 1

        vmax = float(np.nanmax(np.abs(np.asarray(bd.data))))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        self.threshold_min = 0.0
        self.threshold_max = vmax
        self.threshold_step = vmax / 100.0
        self.threshold = float(threshold) if threshold is not None else 0.0

        self._refresh_html()

    @traitlets.observe("volume_idx", "threshold")
    def _on_change(self, _change):
        self._refresh_html()

    def _repr_mimebundle_(self, **kwargs):
        """Add a text/html fallback so static-built docs (mystmd, JB v2) render
        the nilearn ortho/surface viewer when no live widget runtime is
        available. Live runtimes (Jupyter, marimo, JupyterLite) prefer the
        widget renderer and get the full slider experience.

        For 4D BrainData, the fallback pre-renders all volumes at the
        current threshold and embeds a JS slider that swaps the visible
        frame — preserving volume stepping in static-built docs.
        """
        bundle = super()._repr_mimebundle_(**kwargs)
        if isinstance(bundle, tuple):
            data, metadata = bundle
        else:
            data, metadata = bundle, {}
        data = dict(data)
        data["text/html"] = self._build_static_fallback_html()
        return data, metadata

    def _render_one(self, volume_idx: int, threshold: float) -> str:
        """Render the nilearn HTML viewer for a single volume + threshold."""
        sub = self._bd[volume_idx] if self.has_volume_slider else self._bd
        # Clamp threshold to the current volume's vmax so nilearn doesn't warn
        # when stepping into a volume with a smaller dynamic range than the
        # 4D-wide vmax used for the global slider.
        sub_vmax = float(np.nanmax(np.abs(np.asarray(sub.data))))
        thr = max(min(threshold, sub_vmax), 1e-6)
        # Suppress nilearn-internal noise: percentile threshold computation
        # calls np.partition on a masked array (UserWarning), and view_img
        # may emit threshold/cut-coord notices. None are actionable here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self._view == "surface":
                v = view_img_on_surf(sub.to_nifti(), threshold=thr, **self._kwargs)
            else:
                v = view_img(sub.to_nifti(), threshold=thr, **self._kwargs)
            return v.get_standalone()

    def _refresh_html(self) -> None:
        self.html = self._render_one(self.volume_idx, self.threshold)

    def _build_static_fallback_html(self) -> str:
        """Build a self-contained HTML representation for static-doc rendering.

        - 3D: a single iframe holding the current nilearn viewer.
        - 4D: pre-render every volume at the current threshold, embed all
          srcdocs in a JS array, and ship a `<input type="range">`
          volume slider that swaps the iframe's srcdoc on input. Threshold
          stays fixed at the widget's current value (no Python in the loop).

        The whole fallback is wrapped in an outer iframe srcdoc so that
        any embedded `<script>` runs in an isolated document context,
        regardless of how the consumer (Jupyter, mystmd, nbconvert) injects
        the text/html mimetype.
        """
        import json

        if not self.has_volume_slider:
            inner_doc = self.html
            iframe_height = 520
        else:
            srcdocs = [
                self._render_one(i, self.threshold) for i in range(self.n_volumes)
            ]
            n = self.n_volumes
            # Escape `</` to `<\/` so `</script>` literals inside the
            # nilearn HTML don't prematurely terminate the outer <script>
            # tag we're embedding this JSON into.
            srcdocs_json = json.dumps(srcdocs).replace("</", "<\\/")
            srcdoc0_attr = _html_escape_for_attr(srcdocs[0])
            inner_doc = f"""<!DOCTYPE html>
<html><body style="margin:0;font-family:sans-serif">
<div style="display:flex;align-items:center;gap:8px;padding:4px;">
  <label style="min-width:80px;font-size:0.9em;">Volume</label>
  <input type="range" id="bv-slider" min="0" max="{n - 1}" step="1" value="0" style="flex:1"/>
  <span id="bv-readout" style="min-width:80px;font-size:0.85em;color:#555;">0 / {n - 1}</span>
</div>
<iframe id="bv-frame" sandbox="allow-scripts allow-same-origin"
        style="width:100%;height:520px;border:0;"
        srcdoc="{srcdoc0_attr}"></iframe>
<script>
(function() {{
  const docs = {srcdocs_json};
  const slider = document.getElementById('bv-slider');
  const readout = document.getElementById('bv-readout');
  const frame = document.getElementById('bv-frame');
  slider.addEventListener('input', function() {{
    const i = Number(slider.value);
    readout.textContent = i + ' / ' + (docs.length - 1);
    frame.srcdoc = docs[i];
  }});
}})();
</script>
</body></html>"""
            iframe_height = 580  # extra room for slider row

        return (
            f'<iframe srcdoc="{_html_escape_for_attr(inner_doc)}" '
            f'sandbox="allow-scripts allow-same-origin" '
            f'style="width:100%;height:{iframe_height}px;border:0;"></iframe>'
        )
