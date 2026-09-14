"""Contract tests for the maintainer scripts under `scripts/`.

These are not part of the package, so they have no other test home; each test
here pins one behaviour a script gets wrong when left unchecked.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class TestCheckKwonly:
    """The `*`-marker check counts loose kwargs, not the presence of a `*`."""

    def test_defaults_before_the_star_marker_are_flagged(self, tmp_path):
        import check_kwonly

        bad = tmp_path / "bad.py"
        bad.write_text("def f(data, a=1, b=2, c=3, *, extra=None):\n    pass\n")
        good = tmp_path / "good.py"
        good.write_text("def f(data, *, a=1, b=2, c=3, extra=None):\n    pass\n")

        assert len(check_kwonly.check_file(bad, {})) == 1
        assert check_kwonly.check_file(good, {}) == []


class TestDocsShow:
    """A cell's final expression is wrapped in place, not by whole lines."""

    def test_statement_sharing_the_last_line_still_compiles(self):
        import docs_show

        executed, _ = docs_show.transform_cell("x = 1; x")
        compile(executed, "<cell>", "exec")
