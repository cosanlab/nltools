"""Contract tests for the maintainer scripts under `scripts/`.

These are not part of the package, so they have no other test home; each test
here pins one behaviour a script gets wrong when left unchecked.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"

# `scripts/` ships with the repository but not with the sdist, so a suite run
# from an unpacked sdist skips these rather than failing to import.
pytestmark = pytest.mark.skipif(
    not SCRIPTS_DIR.exists(), reason="maintainer scripts are not installed"
)


def load_script(name):
    """Import one `scripts/` module by path, without putting it on `sys.path`."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckKwonly:
    """The `*`-marker check counts loose kwargs, not the presence of a `*`."""

    def test_defaults_before_the_star_marker_are_flagged(self, tmp_path):
        check_kwonly = load_script("check_kwonly")

        bad = tmp_path / "bad.py"
        bad.write_text("def f(data, a=1, b=2, c=3, *, extra=None):\n    pass\n")
        good = tmp_path / "good.py"
        good.write_text("def f(data, *, a=1, b=2, c=3, extra=None):\n    pass\n")

        assert len(check_kwonly.check_file(bad, {})) == 1
        assert check_kwonly.check_file(good, {}) == []


class TestDocsShow:
    """A cell's final expression is wrapped in place, not by whole lines."""

    def test_statement_sharing_the_last_line_still_compiles(self):
        docs_show = load_script("docs_show")

        executed, _ = docs_show.transform_cell("x = 1; x")
        compile(executed, "<cell>", "exec")


class TestReleaseTestState:
    """A pytest log naming an error is not a clean test state."""

    def test_error_counts_are_not_reported_as_passing(self, tmp_path, monkeypatch):
        release = load_script("release")

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


class TestMakeTutorialData:
    """A trim that cuts through a block is refused, not silently dropped."""

    def test_event_straddling_the_trim_raises(self):
        import pandas as pd

        make_tutorial_data = load_script("make_tutorial_data")

        events = pd.DataFrame(
            {
                "onset": [0.0, 2.0, 4.0],
                "duration": [2.0, 2.0, 2.0],
                "trial_type": ["language", "string", "language"],
            }
        )
        with pytest.raises(ValueError, match="cuts through an event"):
            make_tutorial_data.trim_events(events, 5.0, subject="01", volumes=5)

    def test_trim_on_a_block_boundary_keeps_whole_events(self):
        import pandas as pd

        make_tutorial_data = load_script("make_tutorial_data")

        events = pd.DataFrame(
            {
                "onset": [0.0, 2.0, 4.0],
                "duration": [2.0, 2.0, 2.0],
                "trial_type": ["language", "string", "language"],
            }
        )
        kept = make_tutorial_data.trim_events(events, 4.0, subject="01", volumes=4)
        assert list(kept["onset"]) == [0.0, 2.0]
