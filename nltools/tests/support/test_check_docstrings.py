"""Tests for the docstring-convention checker in scripts/check_docstrings.py.

The script isn't a package member, so it's loaded by file path via importlib.
Focus: each rule fires on the exact construct griffe2md mis-renders (with a
correct file:line), stays quiet on the house style, and the exit status
is 1 on any finding.
"""

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "check_docstrings.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_docstrings", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _findings(checker, source: str):
    return checker.check_source(textwrap.dedent(source), "pkg/mod.py")


def _by_category(findings, category):
    return [f for f in findings if f.category == category]


class TestSectionHeaders:
    """Singular or off-style headers griffe treats as plain admonitions."""

    @pytest.mark.parametrize(
        "header", ["Example:", "Return:", "Arg:", "Parameter:", "Parameters:", "Raise:"]
    )
    def test_flags_bad_header(self, checker, header):
        src = f'''
            def f(x):
                """Do a thing.

                {header}
                    body
                """
        '''
        hits = _by_category(_findings(checker, src), "section-header")
        assert len(hits) == 1
        assert hits[0].lineno == 5
        assert header in hits[0].message

    @pytest.mark.parametrize("header", ["Examples:", "Returns:", "Args:", "Raises:"])
    def test_accepts_standard_header(self, checker, header):
        src = f'''
            def f(x):
                """Do a thing.

                {header}
                    body
                """
        '''
        assert _by_category(_findings(checker, src), "section-header") == []

    def test_header_must_be_alone_on_its_line(self, checker):
        # Prose that happens to start with the word is not a header.
        src = '''
            def f(x):
                """Do a thing.

                Example: this sentence is prose, not a section header, and is fine.
                """
        '''
        assert _by_category(_findings(checker, src), "section-header") == []


class TestDoctest:
    def test_flags_prompt_line(self, checker):
        src = '''
            def f(x):
                """Do a thing.

                Examples:
                    >>> f(1)
                    2
                """
        '''
        hits = _by_category(_findings(checker, src), "doctest")
        assert [h.lineno for h in hits] == [6]

    def test_accepts_fenced_block(self, checker):
        src = '''
            def f(x):
                """Do a thing.

                Examples:
                    ```python
                    f(1)  # → 2
                    ```
                """
        '''
        assert _by_category(_findings(checker, src), "doctest") == []


class TestRst:
    @pytest.mark.parametrize(
        "line",
        [
            ":param x: the thing",
            ":returns: a value",
            ":rtype: int",
            "See :class:`Foo` for details.",
            "See :func:`bar` for details.",
            "See :meth:`Foo.bar` for details.",
            ".. note::",
            ".. warning::",
            "For example::",
        ],
    )
    def test_flags_rst_leftover(self, checker, line):
        src = f'''
            def f(x):
                """Do a thing.

                {line}
                """
        '''
        hits = _by_category(_findings(checker, src), "rst")
        assert [h.lineno for h in hits] == [5]

    def test_accepts_slice_syntax_and_fenced_code(self, checker):
        src = '''
            def f(x):
                """Do a thing.

                Uses `x[::2]` internally.

                ```python
                label = "a::"
                ```
                """
        '''
        assert _by_category(_findings(checker, src), "rst") == []


class TestLegacyArgs:
    def test_flags_legacy_entry(self, checker):
        src = '''
            def f(x, y):
                """Do a thing.

                Args:
                    x: (int) the first thing
                    y (int): the second thing
                """
        '''
        hits = _by_category(_findings(checker, src), "legacy-arg")
        assert [h.lineno for h in hits] == [6]

    def test_ignores_raises_section(self, checker):
        src = '''
            def f(x):
                """Do a thing.

                Raises:
                    ValueError: (only) when x is negative.
                """
        '''
        assert _by_category(_findings(checker, src), "legacy-arg") == []

    def test_ignores_continuation_lines(self, checker):
        # A deeper-indented description line that happens to read `Word: (...)`
        # is prose continuing the entry above, not a legacy entry.
        src = '''
            class C:
                """Do a thing.

                Attributes:
                    deltas (np.ndarray): Fitted log-ratios.
                        Shape: (n_spaces, n_targets). deltas = log(gamma / alpha)
                """
        '''
        assert _by_category(_findings(checker, src), "legacy-arg") == []

    def test_section_ends_at_dedent(self, checker):
        src = '''
            def f(x):
                """Do a thing.

                Args:
                    x (int): the first thing

                Note:
                    y: (not an arg) this is prose in another section
                """
        '''
        assert _by_category(_findings(checker, src), "legacy-arg") == []


class TestSummaryLine:
    def test_flags_missing_terminal_punctuation(self, checker):
        src = '''
            def f(x):
                """Do a thing that wraps onto
                the next line.
                """
        '''
        hits = _by_category(_findings(checker, src), "summary")
        assert [h.lineno for h in hits] == [3]

    def test_flags_overlong_summary(self, checker):
        long = "Do " + "a very " * 25 + "long thing."
        src = f'''
            def f(x):
                """{long}"""
        '''
        hits = _by_category(_findings(checker, src), "summary")
        assert len(hits) == 1

    @pytest.mark.parametrize(
        "summary",
        [
            "Do a thing.",
            "Is it a thing?",
            "Do it!",
            "Return `x` (unchanged).",
            "Deprecated: use `g`.",
        ],
    )
    def test_accepts_terminated_summary(self, checker, summary):
        src = f'''
            def f(x):
                """{summary}"""
        '''
        assert _by_category(_findings(checker, src), "summary") == []


class TestCoverage:
    def test_scans_module_class_and_attribute_docstrings(self, checker):
        src = '''
            """Module summary

            .. note::
            """

            X = 1
            """Attribute doc.

            >>> X
            """


            class C:
                """Class doc.

                Example:
                    body
                """

                def m(self):
                    """Method doc.

                    Return:
                        body
                    """
        '''
        cats = sorted(f.category for f in _findings(checker, src))
        assert cats == ["doctest", "rst", "section-header", "section-header", "summary"]

    def test_line_numbers_track_docstring_start(self, checker):
        src = '''
            import os


            def f(x):
                """Do a thing.

                Args:
                    x: (int) legacy
                """
        '''
        (hit,) = _findings(checker, src)
        # dedent keeps the leading blank line: `x: (int)` is the 9th line.
        assert (hit.path, hit.lineno) == ("pkg/mod.py", 9)

    def test_clean_source_has_no_findings(self, checker):
        src = '''
            """Clean module."""


            def f(x):
                """Do a thing.

                Args:
                    x (int): the thing.

                Returns:
                    int: the result.

                Examples:
                    ```python
                    f(1)  # → 2
                    ```
                """
                return x
        '''
        assert _findings(checker, src) == []


class TestMain:
    def _write(self, tmp_path, body):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "mod.py").write_text(textwrap.dedent(body))
        return pkg

    def test_exit_1_on_error(self, checker, tmp_path, capsys):
        pkg = self._write(
            tmp_path,
            '''
            def f(x):
                """Do a thing.

                Example:
                    body
                """
            ''',
        )
        assert checker.main(["check_docstrings.py", str(pkg)]) == 1
        out = capsys.readouterr().out
        assert "mod.py:5" in out
        assert "section-header: 1" in out

    def test_exit_1_on_summary_finding(self, checker, tmp_path):
        pkg = self._write(
            tmp_path,
            '''
            def f(x):
                """Do a thing that wraps onto
                the next line.
                """
            ''',
        )
        assert checker.main(["check_docstrings.py", str(pkg)]) == 1

    def test_exit_0_when_clean(self, checker, tmp_path):
        pkg = self._write(
            tmp_path,
            '''
            def f(x):
                """Do a thing."""
            ''',
        )
        assert checker.main(["check_docstrings.py", str(pkg)]) == 0
