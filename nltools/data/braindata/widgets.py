"""Anywidget-based interactive viewer for BrainData.

`BrainViewerWidget` wraps nilearn's HTML viewers (`view_img`,
`view_img_on_surf`) inside an iframe with a threshold panel (with
symmetric/independent and value/percentile toggles), plus a volume
slider for 4D BrainData. Renders inline in Jupyter, marimo, and
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
_ALLOWED_MODES = ("symmetric", "independent")
_ALLOWED_UNITS = ("value", "percentile")


def _html_escape_for_attr(s: str) -> str:
    """Escape an HTML doc for embedding in an iframe srcdoc attribute."""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _percentile_table(values: np.ndarray) -> list[float]:
    """101-entry percentile lookup table for percentiles 0..100.

    Returns a flat zero-filled table when `values` is empty, so the JS
    slider can still bind without special-casing the no-data axis.
    """
    if values.size == 0:
        return [0.0] * 101
    return [float(np.percentile(values, p)) for p in range(101)]


class BrainViewerWidget(anywidget.AnyWidget):
    """Interactive ortho/surface viewer with threshold panel and (for 4D) volume slider."""

    # Volume stepping (4D)
    volume_idx = traitlets.Int(0).tag(sync=True)
    n_volumes = traitlets.Int(1).tag(sync=True)
    has_volume_slider = traitlets.Bool(False).tag(sync=True)

    # Threshold state
    # `upper` is the symmetric |x|>=upper threshold OR the positive cutoff in
    # independent mode. `lower` is unused in symmetric mode and is the
    # (<=0) negative cutoff in independent mode.
    upper = traitlets.Float(0.0).tag(sync=True)
    lower = traitlets.Float(0.0).tag(sync=True)
    mode_signed = traitlets.Bool(False).tag(sync=True)
    mode_pct = traitlets.Bool(False).tag(sync=True)

    # Data range hints, precomputed once and synced for client-side
    # toggle handling (no Python round-trip when flipping units/mode).
    vmax_abs = traitlets.Float(1.0).tag(sync=True)
    data_min = traitlets.Float(0.0).tag(sync=True)
    data_max = traitlets.Float(0.0).tag(sync=True)
    pct_table_abs = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    pct_table_neg = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    pct_table_pos = traitlets.List(trait=traitlets.Float()).tag(sync=True)

    html = traitlets.Unicode("").tag(sync=True)

    _esm = """
    function pctToValue(pct, table) {
      if (!table || table.length !== 101) return 0;
      if (pct <= 0) return table[0];
      if (pct >= 100) return table[100];
      const i = Math.floor(pct);
      const frac = pct - i;
      return table[i] * (1 - frac) + table[i + 1] * frac;
    }
    function valueToPct(val, table) {
      if (!table || table.length !== 101) return 0;
      if (val <= table[0]) return 0;
      if (val >= table[100]) return 100;
      for (let i = 0; i < 100; i++) {
        if (val <= table[i + 1]) {
          const range = table[i + 1] - table[i];
          if (range <= 0) return i;
          return i + (val - table[i]) / range;
        }
      }
      return 100;
    }
    function fmt(v) {
      const a = Math.abs(v);
      if (a === 0) return "0";
      if (a < 0.01 || a >= 10000) return v.toExponential(2);
      return v.toFixed(2);
    }

    function render({ model, el }) {
      el.innerHTML = `
        <div class="bv-controls">
          <div class="bv-row bv-volume-row" style="display:none">
            <label>Volume</label>
            <input type="range" class="bv-volume-slider" min="0" step="1"/>
            <span class="bv-volume-readout"></span>
          </div>
          <div class="bv-row bv-toggle-row">
            <label>Threshold</label>
            <div class="bv-toggle-group" data-key="units">
              <button type="button" class="bv-pill" data-val="value">Value</button>
              <button type="button" class="bv-pill" data-val="percentile">Percentile</button>
            </div>
            <div class="bv-toggle-group" data-key="mode">
              <button type="button" class="bv-pill" data-val="symmetric">Symmetric</button>
              <button type="button" class="bv-pill" data-val="independent">Independent</button>
            </div>
          </div>
          <div class="bv-row bv-sym-row">
            <label class="bv-thr-label">Threshold</label>
            <input type="range" class="bv-sym-slider"/>
            <span class="bv-sym-readout"></span>
          </div>
          <div class="bv-row bv-neg-row" style="display:none">
            <label class="bv-neg-label">Negative</label>
            <input type="range" class="bv-neg-slider"/>
            <span class="bv-neg-readout"></span>
          </div>
          <div class="bv-row bv-pos-row" style="display:none">
            <label class="bv-pos-label">Positive</label>
            <input type="range" class="bv-pos-slider"/>
            <span class="bv-pos-readout"></span>
          </div>
        </div>
        <iframe class="bv-frame" sandbox="allow-scripts allow-same-origin"></iframe>
      `;

      const $ = (sel) => el.querySelector(sel);
      const volRow = $(".bv-volume-row");
      const volSlider = $(".bv-volume-slider");
      const volReadout = $(".bv-volume-readout");
      const symRow = $(".bv-sym-row");
      const symSlider = $(".bv-sym-slider");
      const symReadout = $(".bv-sym-readout");
      const negRow = $(".bv-neg-row");
      const negSlider = $(".bv-neg-slider");
      const negReadout = $(".bv-neg-readout");
      const posRow = $(".bv-pos-row");
      const posSlider = $(".bv-pos-slider");
      const posReadout = $(".bv-pos-readout");
      const frame = $(".bv-frame");

      // Toggle pill buttons
      el.querySelectorAll(".bv-toggle-group").forEach((group) => {
        group.querySelectorAll(".bv-pill").forEach((pill) => {
          pill.addEventListener("click", () => {
            const key = group.dataset.key;
            const val = pill.dataset.val;
            if (key === "units") {
              model.set("mode_pct", val === "percentile");
            } else if (key === "mode") {
              model.set("mode_signed", val === "independent");
            }
            model.save_changes();
          });
        });
      });

      // Slider configuration & readouts (driven entirely from traitlet state)
      function configureSliders() {
        const pct = model.get("mode_pct");
        const signed = model.get("mode_signed");
        const vmaxAbs = model.get("vmax_abs");
        const dataMin = model.get("data_min");
        const dataMax = model.get("data_max");
        const lowerVal = model.get("lower");
        const upperVal = model.get("upper");

        // Show/hide rows for current mode
        symRow.style.display = signed ? "none" : "";
        negRow.style.display = signed ? "" : "none";
        posRow.style.display = signed ? "" : "none";

        // Reflect active pills
        el.querySelectorAll(".bv-toggle-group").forEach((group) => {
          const key = group.dataset.key;
          const active = key === "units" ? (pct ? "percentile" : "value")
                                         : (signed ? "independent" : "symmetric");
          group.querySelectorAll(".bv-pill").forEach((pill) => {
            pill.classList.toggle("bv-pill-active", pill.dataset.val === active);
          });
        });

        if (!signed) {
          // Symmetric: single slider over [0, vmax_abs] (value) or [0, 100] (pct)
          if (pct) {
            symSlider.min = "0";
            symSlider.max = "100";
            symSlider.step = "0.5";
            const tablePct = valueToPct(upperVal, model.get("pct_table_abs"));
            symSlider.value = String(tablePct);
            symReadout.textContent = `${tablePct.toFixed(1)}%  (|x| ≥ ${fmt(upperVal)})`;
          } else {
            const step = vmaxAbs > 0 ? vmaxAbs / 100 : 0.01;
            symSlider.min = "0";
            symSlider.max = String(vmaxAbs);
            symSlider.step = String(step);
            symSlider.value = String(upperVal);
            symReadout.textContent = `|x| ≥ ${fmt(upperVal)}`;
          }
        } else {
          // Independent: two sliders
          // Negative slider position: data_min (left) ... 0 (right). Larger
          // position = stricter (only very negative pass).
          // Positive slider position: 0 (left) ... data_max (right).
          if (pct) {
            negSlider.min = "0"; negSlider.max = "100"; negSlider.step = "0.5";
            posSlider.min = "0"; posSlider.max = "100"; posSlider.step = "0.5";
            const negPct = valueToPct(Math.abs(lowerVal), model.get("pct_table_neg"));
            const posPct = valueToPct(upperVal, model.get("pct_table_pos"));
            negSlider.value = String(negPct);
            posSlider.value = String(posPct);
            negReadout.textContent = `${negPct.toFixed(1)}%  (x ≤ ${fmt(lowerVal)})`;
            posReadout.textContent = `${posPct.toFixed(1)}%  (x ≥ ${fmt(upperVal)})`;
          } else {
            const negStep = dataMin < 0 ? Math.abs(dataMin) / 100 : 0.01;
            const posStep = dataMax > 0 ? dataMax / 100 : 0.01;
            negSlider.min = String(dataMin); negSlider.max = "0"; negSlider.step = String(negStep);
            posSlider.min = "0"; posSlider.max = String(dataMax); posSlider.step = String(posStep);
            negSlider.value = String(lowerVal);
            posSlider.value = String(upperVal);
            negReadout.textContent = `x ≤ ${fmt(lowerVal)}`;
            posReadout.textContent = `x ≥ ${fmt(upperVal)}`;
          }
          // Disable degenerate sliders if data has no neg/pos side
          negSlider.disabled = !(dataMin < 0);
          posSlider.disabled = !(dataMax > 0);
        }

        // Volume row
        const showVol = model.get("has_volume_slider");
        volRow.style.display = showVol ? "" : "none";
        if (showVol) {
          volSlider.max = String(model.get("n_volumes") - 1);
          volSlider.value = String(model.get("volume_idx"));
          volReadout.textContent = `${model.get("volume_idx")} / ${model.get("n_volumes") - 1}`;
        }
      }

      const syncFrame = () => { frame.srcdoc = model.get("html") || ""; };

      // Live readout updates during drag (no Python round-trip)
      function readoutOnInput(slider, readout, kind) {
        slider.addEventListener("input", () => {
          const pct = model.get("mode_pct");
          const v = Number(slider.value);
          if (kind === "sym") {
            const val = pct ? pctToValue(v, model.get("pct_table_abs")) : v;
            readout.textContent = pct
              ? `${v.toFixed(1)}%  (|x| ≥ ${fmt(val)})`
              : `|x| ≥ ${fmt(val)}`;
          } else if (kind === "neg") {
            const val = pct ? -pctToValue(v, model.get("pct_table_neg")) : v;
            readout.textContent = pct
              ? `${v.toFixed(1)}%  (x ≤ ${fmt(val)})`
              : `x ≤ ${fmt(val)}`;
          } else if (kind === "pos") {
            const val = pct ? pctToValue(v, model.get("pct_table_pos")) : v;
            readout.textContent = pct
              ? `${v.toFixed(1)}%  (x ≥ ${fmt(val)})`
              : `x ≥ ${fmt(val)}`;
          } else if (kind === "vol") {
            readout.textContent = `${v} / ${model.get("n_volumes") - 1}`;
          }
        });
      }
      readoutOnInput(symSlider, symReadout, "sym");
      readoutOnInput(negSlider, negReadout, "neg");
      readoutOnInput(posSlider, posReadout, "pos");
      readoutOnInput(volSlider, volReadout, "vol");

      // Python re-render only on mouse-up ("change") to keep dragging snappy.
      symSlider.addEventListener("change", () => {
        const pct = model.get("mode_pct");
        const v = Number(symSlider.value);
        const val = pct ? pctToValue(v, model.get("pct_table_abs")) : v;
        model.set("upper", val);
        model.save_changes();
      });
      negSlider.addEventListener("change", () => {
        const pct = model.get("mode_pct");
        const v = Number(negSlider.value);
        const val = pct ? -pctToValue(v, model.get("pct_table_neg")) : v;
        model.set("lower", val);
        model.save_changes();
      });
      posSlider.addEventListener("change", () => {
        const pct = model.get("mode_pct");
        const v = Number(posSlider.value);
        const val = pct ? pctToValue(v, model.get("pct_table_pos")) : v;
        model.set("upper", val);
        model.save_changes();
      });
      volSlider.addEventListener("change", () => {
        model.set("volume_idx", Number(volSlider.value));
        model.save_changes();
      });

      model.on("change:html", syncFrame);
      model.on(
        "change:mode_signed change:mode_pct change:lower change:upper " +
        "change:volume_idx change:n_volumes change:has_volume_slider " +
        "change:vmax_abs change:data_min change:data_max",
        configureSliders
      );

      configureSliders();
      syncFrame();
    }
    export default { render };
    """

    _css = """
    .bv-controls { font-family: sans-serif; padding: 4px 0; }
    .bv-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; flex-wrap: wrap; }
    .bv-row label { min-width: 80px; font-size: 0.9em; }
    .bv-row input[type=range] { flex: 1; min-width: 120px; }
    .bv-row span { min-width: 140px; font-size: 0.85em; color: #555; }
    .bv-toggle-group { display: inline-flex; border: 1px solid #ccc; border-radius: 4px; overflow: hidden; }
    .bv-pill {
      padding: 2px 10px; font-size: 0.8em; background: #f6f6f6;
      border: 0; border-right: 1px solid #ccc; cursor: pointer; color: #444;
    }
    .bv-pill:last-child { border-right: 0; }
    .bv-pill-active { background: #4477aa; color: white; }
    .bv-frame { width: 100%; min-height: 520px; border: 0; }
    """

    def __init__(
        self,
        bd,
        *,
        view: str = "ortho",
        mode: str = "symmetric",
        units: str = "value",
        lower: float | None = None,
        upper: float | None = None,
        threshold: float | None = None,
        **view_kwargs,
    ):
        if view not in _ALLOWED_VIEWS:
            raise ValueError(f"view={view!r} not in {_ALLOWED_VIEWS}.")
        if mode not in _ALLOWED_MODES:
            raise ValueError(f"mode={mode!r} not in {_ALLOWED_MODES}.")
        if units not in _ALLOWED_UNITS:
            raise ValueError(f"units={units!r} not in {_ALLOWED_UNITS}.")
        super().__init__()
        self._bd = bd
        self._view = view
        self._kwargs = view_kwargs

        n = bd.shape[0] if len(bd.shape) > 1 else 1
        self.n_volumes = n
        self.has_volume_slider = n > 1

        # Precompute data-range hints from the full BrainData (across all
        # volumes for 4D). Slider ranges stay stable as the user steps
        # through volumes.
        arr = np.asarray(bd.data)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            self.vmax_abs = 1.0
            self.data_min = 0.0
            self.data_max = 0.0
            self.pct_table_abs = [0.0] * 101
            self.pct_table_neg = [0.0] * 101
            self.pct_table_pos = [0.0] * 101
        else:
            abs_vals = np.abs(finite)
            self.vmax_abs = float(abs_vals.max()) or 1.0
            self.data_min = float(min(finite.min(), 0.0))
            self.data_max = float(max(finite.max(), 0.0))
            self.pct_table_abs = _percentile_table(abs_vals)
            self.pct_table_neg = _percentile_table(np.abs(finite[finite < 0]))
            self.pct_table_pos = _percentile_table(finite[finite > 0])

        self.mode_signed = mode == "independent"
        self.mode_pct = units == "percentile"

        # Initial threshold values
        if threshold is not None and upper is None:
            upper = float(threshold)
        self.upper = float(upper) if upper is not None else 0.0
        self.lower = float(lower) if lower is not None else 0.0

        self._refresh_html()

    @traitlets.observe("volume_idx", "upper", "lower", "mode_signed")
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

    def _render_one(self, volume_idx: int) -> str:
        """Render the nilearn HTML viewer for the current threshold state."""
        sub = self._bd[volume_idx] if self.has_volume_slider else self._bd

        if self.mode_signed:
            # We own the masking: zero out voxels where lower < x < upper
            # and pass threshold=0 to nilearn. Lets us implement
            # asymmetric thresholds nilearn doesn't natively support.
            from .utils import shallow_copy

            masked = shallow_copy(sub)
            d = np.asarray(sub.data).copy()
            d[(d > self.lower) & (d < self.upper)] = 0
            masked.data = d
            nifti = masked.to_nifti()
            thr = 0.0
        else:
            nifti = sub.to_nifti()
            thr = float(self.upper)

        # Suppress nilearn-internal noise: percentile threshold computation
        # calls np.partition on a masked array (UserWarning), and view_img
        # may emit threshold/cut-coord notices. None are actionable here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self._view == "surface":
                v = view_img_on_surf(nifti, threshold=thr, **self._kwargs)
            else:
                v = view_img(nifti, threshold=thr, **self._kwargs)
            return v.get_standalone()

    def _refresh_html(self) -> None:
        self.html = self._render_one(self.volume_idx)

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
            srcdocs = [self._render_one(i) for i in range(self.n_volumes)]
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
