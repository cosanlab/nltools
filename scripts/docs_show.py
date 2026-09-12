"""Notebook-style output for the markdown-exec ``python`` cells in the tutorials.

markdown-exec renders only what a cell prints. This module swaps in a ``python``
formatter that behaves like a notebook cell instead:

- the value of the cell's final expression is displayed (DataFrames, models, and
  anything else with ``_repr_html_`` as HTML; other values as text)
- every matplotlib figure the cell created is rendered as inline SVG at the end
  of the cell, then closed (what the inline backend does)
- ``print`` output appears in order, in a preformatted block — including
  what a library the cell calls prints, since stdout is redirected into the
  cell for as long as it runs
- the source shown to readers is the cell exactly as written

It is also the tutorials' build-time gate. A cell that raises fails the build,
and so does a cell that writes to stderr: tutorial cells must not trigger
warnings — fix the analysis, never filter the warning (issue #489). The gate
sees what reaches ``sys.stderr`` while the cell runs, which covers a direct
write and ``warnings.warn``; it cannot see a library that logs through a
``StreamHandler`` that captured the real stream at import time, because
``redirect_stderr`` rebinds the name and not that handler's stream.

``scripts/marimo_to_zensical.py`` opens every generated page with a hidden cell
that activates the formatter for that page and stamps the page with a digest of
its cells::

    ```python exec="on" session="basics-01-brain-data" render="off" setup="on"
    import docs_show

    docs_show.install("docs/tutorials/basics/01_brain_data.py", digest="…", cells=N)
    ```

The formatter stays registered once the first page installed it, so it is what
runs every later page's first cell too: ``setup="on"`` marks that cell, which
runs outside the previous page's record or replay.

Executing a page is what makes a docs build slow, and zensical rebuilds every
page whenever its configuration changes, so each page's outputs are recorded
under ``.tutorial-cache/pages/`` as its cells run. A page whose cells and this
module are unchanged since its record was written replays that record instead
of executing. ``DOCS_EXEC=all`` in the environment executes every page anyway,
which is how ``docs-build`` stays the gate; ``tutorials-clean-cache`` drops the
records together with the notebooks' fit caches.

Every later cell on the page is ``python exec="on" session="<page-slug>"``, with
``source="above"`` unless the notebook hides the source and ``render="off"`` when
the cell should run without appearing at all. The module must be importable when
zensical builds (the docs tasks put ``scripts/`` on ``PYTHONPATH``); when that
import fails, markdown-exec falls back to its own formatter and the pages lose
their outputs and their gate, which is what ``scripts/check_tutorial_pages.py``
catches after the build.

Which notebook is being rendered, and how far into it, are module state, so the
design assumes zensical renders one page at a time in one process — as it does
today. A parallel renderer would scramble the file name and cell number in a
failure message, not the outputs.
"""

from __future__ import annotations

import ast
import hashlib
import html
import io
import linecache
import os
import shutil
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import markdown_exec
import matplotlib
from markdown_exec._internal.formatters import python as markdown_exec_python
from markdown_exec._internal.formatters.base import ExecutionError, base_format
from markdown_exec._internal.formatters.python import _run_python
from pymdownx.superfences import SuperFencesException

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

__all__ = [
    "Cell",
    "CellFailure",
    "exec_with_source",
    "figure_of",
    "format_cell",
    "install",
    "is_open",
    "page_key",
    "render",
    "run_cell",
    "svg_of",
    "transform_cell",
    "unique_line_anchors",
]

# The notebook the page being rendered came from, and how many of its cells have
# run, so a failure names a file the author can open. `install()` sets both from
# the page's first (hidden) cell, which makes the first visible cell number 1.
_notebook = "<unknown notebook>"
_cell_index = 0

# The `Cell` collecting output right now, or None between cells. `_CellStdout`
# reads it to route a library's `print` into the cell that provoked it.
_current_cell: Cell | None = None

# Recorded page outputs, one directory per notebook holding one `<n>.html` per
# cell under a key that changes with the page's cells or with this module.
# Relative to the working directory, like the notebooks' `Memory(".tutorial-cache")`.
PAGE_CACHE_DIR = Path(".tutorial-cache/pages")

# `DOCS_EXEC=all` executes every page even when its record is complete; the
# default executes only pages whose cells changed since they were recorded.
DOCS_EXEC = "DOCS_EXEC"

# The outputs to replay for the page being rendered (a complete record was
# found), or the directory recording this page's outputs as its cells run.
_replay: list[str] | None = None
_record: Path | None = None


class CellFailure(SuperFencesException):
    """A tutorial cell raised or wrote to stderr, which fails the docs build.

    The base class is the one escape hatch out of a fence formatter:
    `pymdownx.superfences` swallows every other exception and renders the fence
    as literal text, and `markdown_exec.ExecutionError` is caught one layer
    earlier still, logged as a warning, and rendered into the page. Only this
    one reaches zensical and stops the build.
    """


def exec_with_source(
    code: str, filename: str, exec_globals: dict | None = None
) -> None:
    """Register a cell's source with `linecache`, then execute it.

    markdown-exec compiles each cell under a synthetic `<code block: …>`
    filename that no file backs, so anything introspecting a function a cell
    defined comes up empty. joblib is the one that matters here: without the
    source it keys a `Memory.cache` entry on a per-process hash, which warns
    about undetectable name collisions and misses the cache on the next build.
    A `linecache` entry with no mtime is how a source-less module makes its
    source available, and `inspect` honours it.
    """
    lines = code.splitlines(keepends=True)
    linecache.cache[filename] = (len(code), None, lines, filename)
    _exec_python(code, filename, exec_globals)


_exec_python = markdown_exec_python.exec_python
markdown_exec_python.exec_python = exec_with_source


def install(
    notebook: str = "<unknown notebook>", digest: str = "", cells: int = 0
) -> None:
    """Register `format_cell` as markdown-exec's `python` formatter for one page.

    A page stamped with a `digest` of its cells replays its recorded outputs
    when the record is complete and neither the cells nor this module changed
    since it was written; otherwise the page executes and is recorded afresh.
    An unstamped page always executes and is never recorded.

    Args:
        notebook: Repo-relative path of the marimo notebook this page is
            generated from, used to name the file in a failure message.
        digest: Digest of the page's executable cells, as stamped by
            `scripts/marimo_to_zensical.py`.
        cells: Number of cells that follow this one on the page.
    """
    global _notebook, _cell_index, _replay, _record
    _notebook = notebook
    _cell_index = 0
    _replay = None
    _record = None
    markdown_exec.formatters["python"] = format_cell
    if not digest or not cells:
        return
    page_dir = PAGE_CACHE_DIR / f"{Path(notebook).parent.name}-{Path(notebook).stem}"
    key_dir = page_dir / page_key(digest, Path(__file__).read_bytes())
    recorded = [key_dir / f"{index}.html" for index in range(1, cells + 1)]
    if os.environ.get(DOCS_EXEC) != "all" and all(p.is_file() for p in recorded):
        _replay = [p.read_text() for p in recorded]
        return
    if page_dir.exists():
        shutil.rmtree(page_dir)
    key_dir.mkdir(parents=True)
    _record = key_dir


def page_key(digest: str, formatter_source: bytes) -> str:
    """Cache key of a page: its cells' digest combined with the formatter's source."""
    return hashlib.sha256(digest.encode() + formatter_source).hexdigest()[:16]


def format_cell(**kwargs: Any) -> str:
    """markdown-exec formatter: run the rewritten cell, show the original source.

    A cell with `render="off"` still runs — its side effects land in the page's
    session and the gate still judges it — but contributes nothing to the page.
    On a page replaying its record, nothing runs: the cell's recorded output is
    returned as it was.
    """
    global _cell_index
    extra = kwargs.setdefault("extra", {})
    setup = _is_on(extra.pop("setup", "off"))
    rendered = not setup and _is_on(extra.pop("render", "on"))
    if _replay is not None and not setup:
        _cell_index += 1
        if _cell_index > len(_replay):
            raise CellFailure(
                f"{_notebook} has more cells than its record; regenerate the page"
            )
        return _replay[_cell_index - 1]
    kwargs["html"] = True
    kwargs["transform_source"] = transform_cell
    output = base_format(language="python", run=run_cell, **kwargs)
    if setup:
        # The page's first cell: it called `install`, which reset the state
        # for this page, and it is neither shown nor recorded.
        return ""
    result = unique_line_anchors(output, _cell_index) if rendered else ""
    if _record is not None:
        (_record / f"{_cell_index}.html").write_text(result)
    return result


def _is_on(value: str) -> bool:
    return value.lower() not in {"0", "no", "off", "false"}


def unique_line_anchors(html_output: str, index: int) -> str:
    """Renumber one cell's code-line anchors so a page's ids stay unique.

    markdown-exec converts every cell through a Markdown instance of its own, so
    `pymdownx.highlight`'s per-document code-block counter restarts at zero and
    each cell on a page emits the same `__codelineno-0-1` ids. The counter is not
    reachable through the fence options, so the ids are renumbered here.
    """
    for marker in ("__codelineno", "__span"):
        html_output = html_output.replace(f"{marker}-0-", f"{marker}-{index}-")
    return html_output


def run_cell(code: str, **kwargs: Any) -> str:
    """Execute one cell, failing the build if it raises or writes to stderr."""
    global _cell_index, _current_cell
    _cell_index += 1
    where = f"{_notebook}, cell {_cell_index}"
    stderr = io.StringIO()
    output = ""
    failure = ""
    try:
        with redirect_stderr(stderr), redirect_stdout(_CellStdout(sys.stdout)):
            output = _run_python(code, **kwargs)
    except ExecutionError as error:
        failure = f"{where} raised:\n\n{unfence(str(error))}"
    else:
        captured = stderr.getvalue()
        if captured.strip():
            failure = (
                f"{where} wrote to stderr:\n\n{captured.rstrip()}\n\n"
                "Tutorial cells must not trigger warnings: fix the analysis, "
                "never filter the warning."
            )
    finally:
        # A cell that raised never reached its `flush`, so clear the binding
        # here rather than leaving a dead cell to catch the next page's output.
        _current_cell = None
    if failure:
        # zensical prints the traceback of an exception raised while rendering
        # but not its message, so the message goes to stderr itself.
        print(f"\ndocs_show: {failure}\n", file=sys.stderr)
        raise CellFailure(failure)
    return output


class _CellStdout(io.TextIOBase):
    """Routes everything written to stdout while a cell runs into that cell.

    `transform_cell` rebinds `print` in the page's globals, which catches what
    the cell itself prints but not what a library it calls prints: a function
    defined in another module resolves `print` through its own globals and lands
    on the real stdout, where it reaches the build log instead of the page.
    Redirecting the stream catches both, in one ordered stream, which is what a
    notebook shows.

    Args:
        fallback: The stream to write to when no cell is running.
    """

    def __init__(self, fallback: Any) -> None:
        self.fallback = fallback

    def write(self, text: str) -> int:
        if _current_cell is None:
            return self.fallback.write(text)
        _current_cell.write(text)
        return len(text)

    def writable(self) -> bool:
        return True


def unfence(text: str) -> str:
    """Strip the markdown code fence markdown-exec wraps a traceback in."""
    lines = text.strip("\n").split("\n")
    if lines and lines[0].startswith("```") and lines[-1].startswith("```"):
        lines = lines[1:-1]
    return "\n".join(lines)


def transform_cell(code: str) -> tuple[str, str]:
    """Return `(code to run, code to display)` for one cell.

    The executed version routes `print` through a `Cell`, wraps the final
    expression statement in `__cell.show(...)`, and flushes pending output
    (including open figures) at the end. The displayed version is the cell
    exactly as written.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, code
    lines = code.split("\n")
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr):
        start, end = last.lineno - 1, last.end_lineno or last.lineno
        # The call opens and closes on lines of its own, so a trailing comment
        # on the expression does not comment out the closing parenthesis.
        lines[start:end] = ["__cell.show(", *lines[start:end], ")"]
    header = [
        "import docs_show as __docs_show",
        "__cell = __docs_show.Cell(print)",
        "print = __cell.print",
    ]
    footer = ["__cell.flush()"]
    return "\n".join(header + lines + footer), code


class Cell:
    """Collects a cell's stdout and rich outputs, emitting HTML in order.

    Construct one at the start of the cell: it records which figures pyplot
    already held, so `flush` can tell the cell's own figures from inherited
    ones.

    Args:
        emit: markdown-exec's buffer-backed `print` for this cell.
    """

    def __init__(self, emit: Callable[..., None]) -> None:
        import matplotlib.pyplot as plt

        global _current_cell

        self.emit = emit
        self.stdout: list[str] = []
        # Anything written to stdout from here on belongs to this cell, whoever
        # wrote it; `run_cell` clears the binding when the cell is done.
        _current_cell = self
        # Figure numbers pyplot was already holding when this cell started. A
        # cell renders what it drew, not what it inherited: the docs build
        # leaves nothing open between cells, but a test on the same pytest-xdist
        # worker can, and that figure is not this cell's output.
        self.inherited: set[int] = set(plt.get_fignums())

    def print(
        self,
        *args: Any,
        sep: str = " ",
        end: str = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        """Collect the cell's `print` output, in order, for the rendered page."""
        text = sep.join(str(arg) for arg in args) + end
        if file is not None:
            # An explicit stream (sys.stderr, an open file) is the author's
            # business, not page output; stderr still fails the build.
            file.write(text)
            if flush:
                file.flush()
            return
        self.stdout.append(text)

    def write(self, text: str) -> None:
        """Collect raw stdout text, in order with the cell's own `print` output."""
        self.stdout.append(text)

    def show(self, obj: Any) -> None:
        """Display the cell's final expression value.

        A figure pyplot still tracks waits for `flush`, which renders every
        open figure in creation order. A figure that is no longer tracked is
        rendered here instead: nltools' plot helpers close the figure they
        created before returning it, so that a live notebook does not draw it
        twice, and `flush` would never see it.
        """
        figure = figure_of(obj)
        if figure is not None:
            if not is_open(figure):
                self.flush_stdout()
                self.emit(svg_of(figure))
            return
        if obj is None:
            return
        self.flush_stdout()
        self.emit(render(obj))

    def flush_stdout(self) -> None:
        """Emit the collected `print` output as one preformatted block."""
        if self.stdout:
            text = "".join(self.stdout).rstrip("\n")
            self.emit(f'<pre class="cell-output">{html.escape(text)}</pre>')
            self.stdout = []

    def flush(self) -> None:
        """Emit pending stdout, then the figures this cell opened, and close them.

        A figure pyplot was already holding when the cell started belongs to
        whoever opened it: it is neither rendered nor closed. Figures the cell
        detached from pyplot were already rendered by `show`.
        """
        import matplotlib.pyplot as plt

        self.flush_stdout()
        for num in plt.get_fignums():
            if num in self.inherited:
                continue
            self.emit(svg_of(plt.figure(num)))
            plt.close(num)


def render(obj: Any) -> str:
    """Return an HTML fragment for `obj`."""
    repr_html = getattr(obj, "_repr_html_", None)
    if callable(repr_html):
        return f'<div class="cell-output">{repr_html()}</div>'
    return f'<pre class="cell-output">{html.escape(repr(obj))}</pre>'


def is_open(fig: matplotlib.figure.Figure) -> bool:
    """Whether pyplot still tracks `fig`, and so will render it at flush.

    `Figure.number` is not the test: `plt.close` hands the number back and a
    later figure can be given it. Closing does drop the figure's manager.
    """
    return getattr(fig.canvas, "manager", None) is not None


def figure_of(obj: Any) -> matplotlib.figure.Figure | None:
    """Extract a matplotlib Figure from a Figure, Axes, or seaborn grid."""
    from matplotlib.figure import Figure

    if isinstance(obj, Figure):
        return obj
    for attr in ("figure", "fig"):
        candidate = getattr(obj, attr, None)
        if isinstance(candidate, Figure):
            return candidate
    return None


def svg_of(fig: matplotlib.figure.Figure) -> str:
    """Serialize a figure to an inline SVG fragment."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    svg = svg[svg.index("<svg") :]
    return f'<div class="cell-output cell-figure">{svg}</div>'


install()
