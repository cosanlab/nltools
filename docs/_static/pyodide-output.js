// Baked output, figures, HTML and widgets in zensical's ```pyodide fences.
//
// Two jobs. On page load, every fence is filled with the output its build-time
// twin produced, so the page reads as finished before anything runs; the first
// Run on a fence clears that and the live output takes over. While a fence runs,
// its value is rendered the way a notebook would render it, so figures, rich
// reprs and anywidget viewers survive instead of collapsing to text.
//
// The seam, and how to re-check it when zensical updates
// -----------------------------------------------------
// Zensical drives the fences itself, from
// `zensical/templates/assets/javascripts/bundle.*.min.js`. Three facts about
// that bundle are what this file hooks into:
//
//   1. It boots Pyodide by calling the *global* `loadPyodide`, and fetches the
//      CDN script only when that global is missing:
//        Js("https://unpkg.com/pyodide@314.0.2/pyodide.js",
//           () => typeof loadPyodide == "undefined" || loadPyodide instanceof Element)
//      Defining the global here therefore hands us the boot, and the Pyodide
//      version stays zensical's: it arrives in the `indexURL` option.
//   2. It writes cell output with `element.textContent = ...`, so nothing it
//      writes can be HTML, but anything appended after the last write survives.
//   3. It appends a cell's value only when that value is neither `null` nor
//      `undefined`, so a wrapper that returns `undefined` renders the value
//      itself without being written over.
//
// So the whole hook is one wrapped method, `runPythonAsync`. Everything else
// stays zensical's: the Ace editor, the lazy boot on the first Run, the
// micropip install, the `data-md-exec-state` attribute and the stdout writer.
//
// Symptoms if a zensical upgrade breaks this: text-only output again (fact 1 no
// longer holds), or a figure that flashes and disappears, or `Figure(400x600)`
// printed under one (facts 2 and 3).
//
// The baked output rides on facts 2 and 3 as well. `scripts/marimo_to_zensical.py`
// emits each twin's output in a `<div class="cell-baked" data-for="N">` before the
// Nth fence on the page; this file moves that div's children into the fence's own
// output element, and zensical's `textContent = ""` on the next run clears them
// with everything else. Symptom if the pairing breaks: baked output under the
// wrong cell, or a page of empty editors.

(() => {
  "use strict";

  // Runs after every cell and returns the HTML it should show. Lives in
  // Pyodide's own globals, which are a different namespace from the
  // per-`session` dicts the fences execute in, so the name never shows up in a
  // reader's session.
  const RENDERER = `
def _nltools_docs_html(value):
    """Return HTML fragments for one cell: its figures, then its value.

    Empty when there is nothing to show as HTML, which is the caller's cue to
    fall back to plain text.
    """
    import io
    import sys

    fragments = []

    def add_figure(figure):
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        svg = buffer.getvalue()
        fragments.append('<div class="pyodide-figure">' + svg[svg.index("<svg") :] + "</div>")

    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        pyplot.rcParams["svg.fonttype"] = "none"
        for number in pyplot.get_fignums():
            add_figure(pyplot.figure(number))
        pyplot.close("all")

    # Every nltools plotting method detaches the figure it drew from pyplot and
    # returns it, so the loop above never sees one; a reader's own pyplot call
    # is the case the loop above is for.
    figure_type = getattr(sys.modules.get("matplotlib.figure"), "Figure", ())
    figures = value if isinstance(value, (list, tuple)) else [value]
    if figures and all(isinstance(figure, figure_type) for figure in figures):
        for figure in figures:
            add_figure(figure)
    elif value is not None:
        repr_html = getattr(value, "_repr_html_", None)
        html = repr_html() if callable(repr_html) else None
        if html:
            fragments.append('<div class="pyodide-html">' + html + "</div>")

    return fragments


def _nltools_docs_widget(value):
    """Describe an anywidget for the browser as esm/css/traits, or None.

    An anywidget carries its frontend in _esm (an ES module, as a string) and
    its state in the traits tagged sync=True. Everything the frontend reads
    goes across here: bytes as a Uint8Array, dicts and lists as plain
    JavaScript objects and arrays. The underscored traits are anywidget's and
    ipywidgets' own plumbing and "layout" is another widget, so neither
    crosses.
    """
    import js
    from pyodide.ffi import to_js

    esm = getattr(value, "_esm", None)
    traits = getattr(value, "traits", None)
    if not isinstance(esm, str) or not callable(traits):
        return None

    state = {}
    for name in traits(sync=True):
        if name.startswith("_"):
            continue
        trait = getattr(value, name, None)
        if hasattr(trait, "traits"):
            continue
        if isinstance(trait, (bytes, bytearray)):
            buffer = js.Uint8Array.new(len(trait))
            if len(trait):
                buffer.assign(trait)
            trait = buffer
        state[name] = trait

    css = getattr(value, "_css", None)
    return to_js(
        {"esm": esm, "css": css if isinstance(css, str) else None, "traits": state},
        dict_converter=js.Object.fromEntries,
    )
`;

  // What a widget's `render` handed back, against the output element it drew
  // into. A fence has no view lifecycle to hang that teardown off, so it is run
  // the next time the same cell runs: without it every Run on a viewer cell
  // strands another WebGL context, and browsers cap how many a page may hold.
  const teardowns = new WeakMap();

  const tearDownWidget = (output) => {
    const teardown = teardowns.get(output);
    if (!teardown) return;
    teardowns.delete(output);
    try {
      teardown();
    } catch (error) {
      // A widget's own teardown failing is its business, not the page's.
      console.warn("widget teardown failed", error);
    }
  };

  // Render one anywidget into a cell's output.
  //
  // A notebook host gives a widget's frontend a live connection to the kernel;
  // a Pyodide fence has nowhere to send a change back to, because the cell's
  // value is gone by the time the reader touches the widget. So the state
  // crosses once, as a snapshot, and this shim stands in for the model: `set`
  // updates the snapshot and fires the frontend's own `change:` listeners, so
  // a widget that drives itself through its model (as anywidget's documented
  // API has it) stays interactive, and `save_changes` has nothing to do.
  const renderWidget = async (widget, output) => {
    const state = widget.traits;
    const listeners = new Map();
    const model = {
      get: (name) => state[name] ?? null,
      set: (name, value) => {
        state[name] = value;
        for (const callback of listeners.get(`change:${name}`) ?? []) callback();
      },
      on: (event, callback) => {
        if (!listeners.has(event)) listeners.set(event, []);
        listeners.get(event).push(callback);
      },
      off: (event, callback) => {
        const bucket = listeners.get(event) ?? [];
        listeners.set(event, callback ? bucket.filter((c) => c !== callback) : []);
      },
      save_changes: () => {},
      send: () => {},
    };

    if (widget.css) {
      const style = document.createElement("style");
      style.textContent = widget.css;
      output.appendChild(style);
    }
    const el = document.createElement("div");
    el.className = "pyodide-widget";
    output.appendChild(el);

    // `_esm` is source, not a file: a Blob URL is what gives the browser
    // something to `import`, and it can be revoked as soon as that resolves.
    const url = URL.createObjectURL(
      new Blob([widget.esm], { type: "text/javascript" }),
    );
    let module;
    try {
      module = await import(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const render = module.default?.render ?? module.render;
    if (typeof render !== "function") {
      throw new Error("widget module exports no render()");
    }
    const teardown = await render({ model, el });
    if (typeof teardown === "function") teardowns.set(output, teardown);
  };

  // Which block is running. Zensical's Run handler and its Ctrl-Enter binding
  // both go through a click on the Run control, and this capture-phase
  // listener on the document sees that click before the handler starts.
  let runningBlock = null;
  document.addEventListener(
    "click",
    (event) => {
      const control = event.target.closest?.("[id$='--run']");
      if (control) runningBlock = control.closest(".pyodide");
    },
    true,
  );

  // Pyodide converts a cell's value to a JavaScript primitive when it can and
  // hands over a proxy of the Python object when it cannot. Only a proxy is
  // worth passing back into Python, and only a proxy has to be freed.
  const isPythonObject = (value) => typeof value?.destroy === "function";

  window.loadPyodide = async (options) => {
    const { loadPyodide } = await import(`${options.indexURL}pyodide.mjs`);
    const pyodide = await loadPyodide(options);
    const runPython = pyodide.runPythonAsync.bind(pyodide);
    let render = null;
    let describeWidget = null;

    pyodide.runPythonAsync = async (code, runOptions) => {
      const output = runningBlock?.querySelector("[id$='--output']");
      if (!output) return runPython(code, runOptions);

      // Zensical clears the output only on the very first run of a block; its
      // stdout writer replaces the text but not the nodes appended below.
      tearDownWidget(output);
      output.textContent = "";

      let value;
      try {
        value = await runPython(code, runOptions);
      } catch (error) {
        // Drop any half-drawn figure so it cannot surface under the next cell.
        render?.(null).destroy();
        throw error;
      }

      if (render === null) {
        await runPython(RENDERER);
        render = pyodide.globals.get("_nltools_docs_html");
        describeWidget = pyodide.globals.get("_nltools_docs_widget");
      }
      const rendered = render(isPythonObject(value) ? value : null);
      const fragments = rendered.toJs();
      rendered.destroy();
      for (const fragment of fragments) output.insertAdjacentHTML("beforeend", fragment);

      let shown = fragments.length > 0;
      if (!shown && isPythonObject(value)) {
        // A widget has no `_repr_html_` to fall back on, so it is checked for
        // here, between the HTML fragments and the plain-text repr. Anything
        // that goes wrong is reported in the cell rather than thrown: a broken
        // widget must not take the rest of the page's Run handling with it.
        try {
          const widget = describeWidget(value);
          if (widget) {
            shown = true;
            await renderWidget(widget, output);
          }
        } catch (error) {
          shown = true;
          output.appendChild(document.createTextNode(`${error}\n`));
        }
      }
      if (!shown && value !== undefined && value !== null) {
        output.appendChild(document.createTextNode(`${String(value)}\n`));
      }
      if (isPythonObject(value)) value.destroy();
      return undefined;
    };

    return pyodide;
  };

  // Fill each fence with its build-time output. The Nth `.pyodide` block on the
  // page takes the baked div marked `data-for="N"`, which the converter numbered
  // in the same order. Moving the nodes (rather than copying the HTML) keeps the
  // inline SVG figures intact, and emptying the div leaves nothing behind to
  // render twice.
  const showBakedOutput = () => {
    document.querySelectorAll(".pyodide").forEach((block, index) => {
      const baked = document.querySelector(`.cell-baked[data-for="${index + 1}"]`);
      const output = block.querySelector("[id$='--output']");
      if (!baked || !output) return;
      while (baked.firstChild) output.appendChild(baked.firstChild);
      baked.remove();
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", showBakedOutput);
  } else {
    showBakedOutput();
  }
})();
