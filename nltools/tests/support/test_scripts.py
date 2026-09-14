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


class TestReleaseTestState:
    """A pytest log naming an error is not a clean test state."""

    def test_error_counts_are_not_reported_as_passing(self, tmp_path, monkeypatch):
        import release

        log = tmp_path / "pytest.log"
        log.write_text("collected 11 items\n\n=== 10 passed, 1 error in 0.1s ===\n")
        assert release.parse_pytest_summary(log)[1] == 1

        monkeypatch.setattr(release, "TEST_LOG", log)
        # Decline the offered re-run, then continue past the warning.
        monkeypatch.setattr(
            release,
            "confirm",
            lambda prompt: "Run the current default suite" not in prompt,
        )
        assert release.step_review_test_state() == "failed"
