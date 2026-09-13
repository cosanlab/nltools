// Figures and HTML in the output of zensical's ```pyodide fences.
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
`;

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

    pyodide.runPythonAsync = async (code, runOptions) => {
      const output = runningBlock?.querySelector("[id$='--output']");
      if (!output) return runPython(code, runOptions);

      // Zensical clears the output only on the very first run of a block; its
      // stdout writer replaces the text but not the nodes appended below.
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
      }
      const rendered = render(isPythonObject(value) ? value : null);
      const fragments = rendered.toJs();
      rendered.destroy();
      for (const fragment of fragments) output.insertAdjacentHTML("beforeend", fragment);

      if (fragments.length === 0 && value !== undefined && value !== null) {
        output.appendChild(document.createTextNode(`${String(value)}\n`));
      }
      if (isPythonObject(value)) value.destroy();
      return undefined;
    };

    return pyodide;
  };
})();
