// niivue viewer for BrainData.iplot(), driven through anywidget's *standard*
// model API (get / set / save_changes / on) only — no host-specific protocol
// (the contrast with ipyniivue, see cosanlab/nltools#455), so the widget works
// identically in Jupyter and `marimo edit`.
//
// niivue itself is pulled from a CDN as an ES module. `_esm` is an ES module, so
// a top-level `import` is resolved by the browser at render time. The pin keeps
// the API stable.
import {
  Niivue,
  NVImage,
  SLICE_TYPE,
} from "https://esm.sh/@niivue/niivue@0.69.0";

// A traitlets `Bytes` trait arrives in JS as a DataView over a shared buffer.
// niivue wants a standalone ArrayBuffer, so slice out exactly this view's
// window. Empty bytes (an absent optional volume) yield `null`.
function bytesToArrayBuffer(dv) {
  if (!dv || dv.byteLength === 0) return null;
  return dv.buffer.slice(dv.byteOffset, dv.byteOffset + dv.byteLength);
}

// NVImage.new treats NaN cal_min/cal_max as "auto" (percentile-derived). A null
// trait (window left on auto) maps to NaN; a number passes through.
function orNaN(x) {
  return x === null || x === undefined ? NaN : x;
}

function formatValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return Number.parseFloat(number.toPrecision(4)).toString();
}

export default {
  async render({ model, el }) {
    el.style.width = "100%";

    // --- DOM: optional threshold controls + the WebGL canvas -------------- //
    let floorInput = null;
    let ceilInput = null;
    let floorValue = null;
    let ceilValue = null;
    if (model.get("controls")) {
      const bounds = model.get("slider_bounds") || {};
      const controls = document.createElement("div");
      controls.style.cssText =
        "display:flex;gap:0.75rem;align-items:center;font:12px/1.4 system-ui," +
        "sans-serif;padding:4px 2px;flex-wrap:wrap";

      const mkRange = (labelText, value) => {
        const wrap = document.createElement("label");
        wrap.style.cssText =
          "display:flex;gap:0.4rem;align-items:center;flex:1 1 200px";
        const span = document.createElement("span");
        span.textContent = labelText;
        const low = document.createElement("span");
        low.textContent = formatValue(bounds.min ?? 0);
        const high = document.createElement("span");
        high.textContent = formatValue(bounds.max ?? 1);
        const input = document.createElement("input");
        input.type = "range";
        input.min = bounds.min ?? 0;
        input.max = bounds.max ?? 1;
        input.step = bounds.step ?? 0.01;
        input.value = value ?? bounds.min ?? 0;
        input.style.flex = "1";
        const output = document.createElement("output");
        output.textContent = formatValue(input.value);
        output.style.cssText = "min-width:6ch;text-align:right;font-variant-numeric:tabular-nums";
        wrap.append(span, low, input, high, output);
        controls.appendChild(wrap);
        return [input, output];
      };

      [floorInput, floorValue] = mkRange("min", bounds.value_low);
      [ceilInput, ceilValue] = mkRange("max", bounds.value_high);
      el.appendChild(controls);
    }

    const height = model.get("height") || 400;
    // niivue's resize observer sizes the canvas from its *parent* and rewrites
    // the canvas's own inline height to 100%, so the requested height has to
    // live on a wrapper. Without it the viewer collapses to the parent's
    // natural height (~150px) in any host that doesn't size the widget for
    // us — static/exported pages, iframes, plain Jupyter output areas.
    const canvasWrap = document.createElement("div");
    canvasWrap.style.cssText = `width:100%;height:${height}px`;
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;height:100%;display:block";
    canvasWrap.appendChild(canvas);
    el.appendChild(canvasWrap);

    // --- niivue instance --------------------------------------------------- //
    const nv = new Niivue({
      isColorbar: model.get("colorbar"),
      ...(model.get("niivue_opts") || {}),
    });
    await nv.attachToCanvas(canvas);

    // Index of the stat map inside nv.volumes (set by loadStack); the threshold
    // controls drive this volume's window.
    let statmapIdx = -1;

    // Build the volume stack [background?, statmap, atlas?] from the byte
    // traits + display params. Rebuilt whenever any volume's bytes change.
    async function loadStack() {
      while (nv.volumes.length) nv.removeVolumeByIndex(0);
      statmapIdx = -1;

      const bg = bytesToArrayBuffer(model.get("bg_bytes"));
      if (bg) {
        const bgVol = await NVImage.new(bg, "background.nii.gz", "gray");
        bgVol.colorbarVisible = false;
        nv.addVolume(bgVol);
      }

      const stat = bytesToArrayBuffer(model.get("statmap_bytes"));
      if (stat) {
        const p = model.get("statmap") || {};
        const vol = await NVImage.new(
          stat,
          (p.name || "statmap") + ".nii.gz",
          p.colormap || "warm",
          p.opacity ?? 1.0,
          null,
          orNaN(model.get("cal_min")),
          orNaN(model.get("cal_max")),
        );
        if (p.colormap_negative) vol.colormapNegative = p.colormap_negative;
        vol.cal_minNeg = orNaN(model.get("cal_min_neg"));
        vol.cal_maxNeg = orNaN(model.get("cal_max_neg"));
        vol.colorbarVisible = true;
        statmapIdx = nv.volumes.length;
        nv.addVolume(vol);
      }

      const atlas = bytesToArrayBuffer(model.get("atlas_bytes"));
      if (atlas) {
        const p = model.get("statmap") || {};
        const vol = await NVImage.new(
          atlas,
          (model.get("atlas_name") || "atlas") + ".nii.gz",
          "",
          p.opacity ?? 1.0,
        );
        const lut = model.get("atlas_lut");
        if (lut && lut.labels) vol.setColormapLabel(lut);
        vol.colorbarVisible = false;
        nv.addVolume(vol);
      }

      nv.opts.atlasOutline = model.get("atlas_outline") || 0;
      nv.updateGLVolume();
    }

    await loadStack();
    nv.setSliceType(SLICE_TYPE[model.get("slice_type")] ?? SLICE_TYPE.MULTIPLANAR);

    // --- threshold controls -> niivue + Python ---------------------------- //
    function applyWindow() {
      if (statmapIdx < 0) return;
      const vol = nv.volumes[statmapIdx];
      const lo = floorInput ? parseFloat(floorInput.value) : model.get("cal_min");
      const hi = ceilInput ? parseFloat(ceilInput.value) : model.get("cal_max");
      if (lo !== null && lo !== undefined && !Number.isNaN(lo)) vol.cal_min = lo;
      if (hi !== null && hi !== undefined && !Number.isNaN(hi)) vol.cal_max = hi;
      if (model.get("mirror_negative")) {
        if (hi !== null && hi !== undefined && !Number.isNaN(hi)) vol.cal_minNeg = -hi;
        if (lo !== null && lo !== undefined && !Number.isNaN(lo)) vol.cal_maxNeg = -lo;
      } else if (lo !== null && lo !== undefined && !Number.isNaN(lo)) {
        vol.cal_maxNeg = -lo;
      }
      nv.updateGLVolume();
    }

    if (floorInput) {
      floorInput.addEventListener("input", () => {
        floorValue.textContent = formatValue(floorInput.value);
        applyWindow();
        model.set("cal_min", parseFloat(floorInput.value));
        model.save_changes();
      });
    }
    if (ceilInput) {
      ceilInput.addEventListener("input", () => {
        ceilValue.textContent = formatValue(ceilInput.value);
        applyWindow();
        model.set("cal_max", parseFloat(ceilInput.value));
        model.save_changes();
      });
    }

    // --- Python -> JS reactivity ------------------------------------------ //
    const onBytes = () => loadStack();
    model.on("change:statmap_bytes", onBytes);
    model.on("change:bg_bytes", onBytes);
    model.on("change:atlas_bytes", onBytes);

    model.on("change:cal_min", () => {
      if (floorInput) floorInput.value = model.get("cal_min") ?? floorInput.value;
      if (floorValue) floorValue.textContent = formatValue(floorInput.value);
      applyWindow();
    });
    model.on("change:cal_max", () => {
      if (ceilInput) ceilInput.value = model.get("cal_max") ?? ceilInput.value;
      if (ceilValue) ceilValue.textContent = formatValue(ceilInput.value);
      applyWindow();
    });
    model.on("change:slice_type", () =>
      nv.setSliceType(SLICE_TYPE[model.get("slice_type")] ?? SLICE_TYPE.MULTIPLANAR),
    );
    model.on("change:colorbar", () => {
      nv.opts.isColorbar = model.get("colorbar");
      nv.drawScene();
    });
    model.on("change:atlas_outline", () => {
      nv.opts.atlasOutline = model.get("atlas_outline") || 0;
      nv.drawScene();
    });

    // niivue has no destroy(); drop the WebGL context on teardown / re-render
    // so repeated cell runs don't leak GL contexts (browsers cap them).
    return () => {
      canvas
        .getContext("webgl2")
        ?.getExtension("WEBGL_lose_context")
        ?.loseContext();
    };
  },
};
