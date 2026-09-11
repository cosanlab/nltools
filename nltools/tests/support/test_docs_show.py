"""Tests for the tutorial cell formatter in scripts/docs_show.py.

The script isn't a package member, so it's loaded by file path via importlib.
Focus: the source rewriting that turns a markdown-exec cell into a notebook
cell, the output collector, and the build-time gate that fails a build on a
cell that raises or writes to stderr.
"""

import importlib.util
import linecache
import sys
from pathlib import Path

import matplotlib
import pytest
from pymdownx.superfences import SuperFencesException

_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "docs_show.py"


@pytest.fixture(scope="module")
def docs_show():
    spec = importlib.util.spec_from_file_location("docs_show", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # The rewritten cells import the module by name while they run.
    sys.modules["docs_show"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def page(docs_show):
    """A page's worth of state: the formatter installed, the counter at zero."""
    docs_show.install("docs/tutorials/basics/01_brain_data.py")
    return docs_show


class Collector:
    """Stands in for markdown-exec's buffer-backed `print`."""

    def __init__(self):
        self.chunks = []

    def __call__(self, text):
        self.chunks.append(text)

    @property
    def html(self):
        return "".join(self.chunks)


class TestTransformCell:
    """`(code to run, code to display)` for one cell."""

    def test_displayed_source_is_the_cell_as_written(self, docs_show):
        code = "x = 1\nx + 1"
        assert docs_show.transform_cell(code)[1] == code

    def test_final_expression_is_displayed(self, docs_show):
        run, _ = docs_show.transform_cell("x = 1\nx + 1")
        assert "__cell.show(\nx + 1\n)" in run

    def test_trailing_comment_does_not_swallow_the_closing_paren(self, docs_show):
        run, _ = docs_show.transform_cell("data.shape  # (images, voxels)")
        compile(run, "<test>", "exec")

    def test_multiline_final_expression_is_wrapped_whole(self, docs_show):
        run, _ = docs_show.transform_cell("plot(\n    x,\n    y,\n)")
        assert "__cell.show(\nplot(\n    x,\n    y,\n)\n)" in run

    def test_final_statement_that_is_not_an_expression_is_left_alone(self, docs_show):
        run, _ = docs_show.transform_cell("x = 1")
        assert "__cell.show" not in run

    def test_print_is_routed_through_the_cell(self, docs_show):
        run, _ = docs_show.transform_cell("print('hi')")
        assert "__cell = __docs_show.Cell(print)" in run
        assert "print = __cell.print" in run

    def test_pending_output_is_flushed_at_the_end(self, docs_show):
        run, _ = docs_show.transform_cell("x = 1")
        assert run.endswith("__cell.flush()")

    def test_a_cell_that_does_not_parse_is_returned_unchanged(self, docs_show):
        code = "def broken("
        assert docs_show.transform_cell(code) == (code, code)


class TestCell:
    """The per-cell output collector."""

    def test_print_output_is_one_preformatted_block(self, docs_show):
        out = Collector()
        cell = docs_show.Cell(out)
        cell.print("one")
        cell.print("two", 3)
        cell.flush()
        assert out.html == '<pre class="cell-output">one\ntwo 3</pre>'

    def test_print_escapes_html(self, docs_show):
        out = Collector()
        cell = docs_show.Cell(out)
        cell.print("<b>")
        cell.flush()
        assert "&lt;b&gt;" in out.html

    def test_print_to_an_explicit_stream_is_not_page_output(self, docs_show, capsys):
        out = Collector()
        cell = docs_show.Cell(out)
        cell.print("to stderr", file=sys.stderr)
        cell.flush()
        assert out.html == ""
        assert capsys.readouterr().err == "to stderr\n"

    def test_printed_text_precedes_the_displayed_value(self, docs_show):
        out = Collector()
        cell = docs_show.Cell(out)
        cell.print("first")
        cell.show("second")
        cell.flush()
        assert out.html.index("first") < out.html.index("second")

    def test_a_final_expression_of_none_shows_nothing(self, docs_show):
        out = Collector()
        cell = docs_show.Cell(out)
        cell.show(None)
        cell.flush()
        assert out.html == ""

    def test_figures_are_emitted_and_closed_at_the_end(self, docs_show):
        import matplotlib.pyplot as plt

        plt.close("all")
        out = Collector()
        cell = docs_show.Cell(out)
        fig, ax = plt.subplots()
        cell.show(ax)
        assert out.html == ""  # the figure waits for the flush
        cell.flush()
        assert out.html.count('class="cell-output cell-figure"') == 1
        assert plt.get_fignums() == []

    def test_a_figure_detached_from_pyplot_is_still_rendered(self, docs_show):
        """nltools' plot helpers close their figure and return it.

        `flush` only sees figures pyplot still tracks, so a returned-but-closed
        figure has to be rendered when it is shown or it is lost.
        """
        import matplotlib.pyplot as plt

        plt.close("all")
        out = Collector()
        cell = docs_show.Cell(out)
        fig, _ = plt.subplots()
        plt.close(fig)

        cell.show(fig)
        assert out.html.count('class="cell-output cell-figure"') == 1
        cell.flush()
        assert out.html.count('class="cell-output cell-figure"') == 1

    def test_a_figure_open_before_the_cell_is_not_rendered_or_closed(self, docs_show):
        """A cell renders the figures it made, not the ones it inherited.

        Under `pytest -n`, an unrelated test on the same worker can leave a
        figure open; before, `flush` swept it into this cell's output.
        """
        import matplotlib.pyplot as plt

        plt.close("all")
        inherited, _ = plt.subplots()

        out = Collector()
        cell = docs_show.Cell(out)
        cell.print("only text")
        cell.flush()

        assert out.html == '<pre class="cell-output">only text</pre>'
        assert plt.get_fignums() == [inherited.number]
        plt.close("all")

    def test_an_inherited_figure_survives_a_cell_that_draws(self, docs_show):
        import matplotlib.pyplot as plt

        plt.close("all")
        inherited, _ = plt.subplots()

        out = Collector()
        cell = docs_show.Cell(out)
        mine, _ = plt.subplots()
        cell.flush()

        assert out.html.count('class="cell-output cell-figure"') == 1
        assert plt.get_fignums() == [inherited.number]
        assert mine.number not in plt.get_fignums()
        plt.close("all")

    def test_a_detached_figure_follows_the_text_the_cell_printed(self, docs_show):
        import matplotlib.pyplot as plt

        plt.close("all")
        out = Collector()
        cell = docs_show.Cell(out)
        fig, _ = plt.subplots()
        plt.close(fig)

        cell.print("first")
        cell.show(fig)
        cell.flush()
        assert out.html.index("first") < out.html.index("cell-figure")


class TestRender:
    """How a displayed value becomes HTML."""

    def test_repr_html_is_used_as_html(self, docs_show):
        class Table:
            def _repr_html_(self):
                return "<table></table>"

        assert (
            docs_show.render(Table())
            == '<div class="cell-output"><table></table></div>'
        )

    def test_other_values_are_escaped_text(self, docs_show):
        assert (
            docs_show.render("<b>")
            == '<pre class="cell-output">&#x27;&lt;b&gt;&#x27;</pre>'
        )


class TestFigures:
    def test_a_figure_is_its_own_figure(self, docs_show):
        fig = matplotlib.figure.Figure()
        assert docs_show.figure_of(fig) is fig

    def test_an_axes_resolves_to_its_figure(self, docs_show):
        fig = matplotlib.figure.Figure()
        assert docs_show.figure_of(fig.add_subplot()) is fig

    def test_a_seaborn_grid_resolves_through_fig(self, docs_show):
        fig = matplotlib.figure.Figure()
        grid = type("Grid", (), {"fig": fig})()
        assert docs_show.figure_of(grid) is fig

    def test_a_plain_value_has_no_figure(self, docs_show):
        assert docs_show.figure_of(42) is None

    def test_a_figure_serializes_to_inline_svg(self, docs_show):
        fig = matplotlib.figure.Figure()
        fig.add_subplot().plot([0, 1], [1, 0])
        svg = docs_show.svg_of(fig)
        assert svg.startswith('<div class="cell-output cell-figure"><svg')
        assert svg.endswith("</div>")


class TestBuildGate:
    """A cell that raises or warns must stop the build."""

    def test_a_clean_cell_returns_its_printed_output(self, page):
        assert page.run_cell("print('hi')", session="basics-01-brain-data") == "hi\n"

    def test_a_raising_cell_fails_the_build(self, page):
        with pytest.raises(page.CellFailure) as failure:
            page.run_cell("1 / 0", session="basics-01-brain-data")
        message = str(failure.value)
        assert "docs/tutorials/basics/01_brain_data.py, cell 1 raised" in message
        assert "ZeroDivisionError" in message
        assert "```" not in message

    def test_a_cell_that_writes_to_stderr_fails_the_build(self, page):
        # A warning reaches stderr through `warnings.showwarning`, which pytest
        # replaces for the duration of a test, so the cell writes the stream
        # directly. The build transcripts cover the warning itself.
        code = "import sys\n\nsys.stderr.write('a stray warning')"
        with pytest.raises(page.CellFailure) as failure:
            page.run_cell(code, session="basics-01-brain-data")
        message = str(failure.value)
        assert (
            "docs/tutorials/basics/01_brain_data.py, cell 1 wrote to stderr" in message
        )
        assert "a stray warning" in message
        assert "fix the analysis, never filter the warning" in message

    def test_the_cell_number_counts_from_the_first_visible_cell(self, page):
        page.run_cell("x = 1", session="basics-01-brain-data")
        with pytest.raises(page.CellFailure) as failure:
            page.run_cell("1 / 0", session="basics-01-brain-data")
        assert "cell 2 raised" in str(failure.value)

    def test_a_failure_escapes_superfences(self, docs_show):
        # Every other exception type is swallowed by pymdownx.superfences,
        # which then renders the fence as literal text and builds happily.
        assert issubclass(docs_show.CellFailure, SuperFencesException)

    def test_stderr_is_restored_after_a_cell(self, page, capsys):
        page.run_cell("x = 1", session="basics-01-brain-data")
        sys.stderr.write("outside a cell")
        assert "outside a cell" in capsys.readouterr().err


class TestSourceRegistration:
    """Cells must be introspectable, or joblib re-fits on every build."""

    def test_a_cell_registers_its_source_with_linecache(self, docs_show):
        filename = "<code block: test>"
        linecache.cache.pop(filename, None)
        docs_show.exec_with_source("x = 1\n", filename, {})
        assert "".join(linecache.cache[filename][2]) == "x = 1\n"

    def test_markdown_exec_executes_through_the_patched_function(self, docs_show):
        # The patch is what puts the source in linecache; if markdown-exec ever
        # resolves `exec_python` differently, joblib silently re-fits forever.
        from markdown_exec._internal.formatters import python as markdown_exec_python

        assert markdown_exec_python.exec_python is docs_show.exec_with_source

    def test_a_function_defined_in_a_cell_has_findable_source(self, docs_show):
        import inspect

        namespace = {}
        docs_show.exec_with_source(
            "def fit(n):\n    return n\n", "<code block: test fit>", namespace
        )
        assert inspect.getsource(namespace["fit"]) == "def fit(n):\n    return n\n"


class TestHiddenCell:
    """`render="off"`: the cell runs, and nothing of it reaches the page."""

    def test_a_hidden_cell_contributes_nothing_to_the_page(self, page):
        assert (
            page.format_cell(
                code="print('not on the page')",
                md=None,
                session="basics-01-brain-data",
                extra={"render": "off"},
            )
            == ""
        )

    def test_the_same_cell_shown_does_reach_the_page(self, page):
        output = page.format_cell(
            code="print('on the page')", md=None, session="basics-01-brain-data"
        )
        assert "on the page" in output

    def test_a_hidden_cell_still_runs(self, page):
        page.format_cell(
            code="hidden_value = 41",
            md=None,
            session="basics-01-brain-data",
            extra={"render": "off"},
        )
        output = page.format_cell(
            code="print(hidden_value + 1)", md=None, session="basics-01-brain-data"
        )
        assert "42" in output

    def test_a_hidden_cell_is_still_gated(self, page):
        with pytest.raises(page.CellFailure):
            page.format_cell(
                code="import sys\n\nsys.stderr.write('warned')",
                md=None,
                session="basics-01-brain-data",
                extra={"render": "off"},
            )


class TestLineAnchors:
    """Every cell's code block would otherwise carry the same HTML ids."""

    def test_anchors_are_renumbered_per_cell(self, docs_show):
        block = '<a id="__codelineno-0-1" href="#__codelineno-0-1"></a>'
        assert docs_show.unique_line_anchors(block, 3) == (
            '<a id="__codelineno-3-1" href="#__codelineno-3-1"></a>'
        )

    def test_line_spans_are_renumbered_too(self, docs_show):
        assert docs_show.unique_line_anchors('<span id="__span-0-2">', 7) == (
            '<span id="__span-0-2">'.replace("-0-", "-7-")
        )

    def test_cell_output_is_left_alone(self, docs_show):
        html = '<pre class="cell-output">0-1</pre>'
        assert docs_show.unique_line_anchors(html, 3) == html


class TestInstall:
    def test_install_registers_the_formatter(self, docs_show):
        import markdown_exec

        docs_show.install("docs/tutorials/basics/01_brain_data.py")
        assert markdown_exec.formatters["python"] is docs_show.format_cell
