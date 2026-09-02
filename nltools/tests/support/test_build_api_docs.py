"""Tests for scripts/build_api_docs.py (the griffe2md driver).

The script isn't a package member, so it's loaded by file path via importlib.
Focus: the parts that don't need griffe2md to run — warning surfacing, the
docs/api drift check, and the page registry's agreement with the site TOC.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).parents[3]
_SCRIPT = _ROOT / "scripts" / "build_api_docs.py"


@pytest.fixture(scope="module")
def build_mod():
    sys.path.insert(0, str(_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("build_api_docs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolve `from __future__` annotations through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestWarningReport:
    """griffe warnings are surfaced deduplicated, grouped by file, with a count.

    griffe2md exits 0 while printing warnings to stderr, so the old driver
    (which only read stderr on a non-zero exit) swallowed every one. The same
    object rendered on several pages (a facade class and its own page) repeats
    its warnings; the report shows each once.
    """

    STDERR = (
        "nltools/data/braindata/__init__.py:30: No type or annotation for parameter 'data'\n"
        "nltools/data/braindata/__init__.py:40: No type or annotation for parameter 'mask'\n"
        "nltools/algorithms/corrections.py:19: No type or annotation for parameter 'p'\n"
    )

    def test_parse_keeps_annotation_warnings(self, build_mod):
        # "No type or annotation" lines are real docstring bugs — never filtered.
        warnings = build_mod.parse_warnings(self.STDERR)
        assert len(warnings) == 3
        assert all("No type or annotation" in w for w in warnings)

    def test_report_dedupes_groups_and_counts(self, build_mod):
        warnings = build_mod.parse_warnings(self.STDERR + self.STDERR)
        report = build_mod.warning_report(warnings)
        assert report.count("parameter 'data'") == 1
        # Grouped under a file header, files in path order.
        assert report.index("nltools/algorithms/corrections.py") < report.index(
            "nltools/data/braindata/__init__.py"
        )
        assert "3 griffe warnings in 2 files" in report

    def test_empty_report(self, build_mod):
        assert build_mod.warning_report(set()) == "0 griffe warnings"


class TestDriftCheck:
    """`--check` regenerates into a temp dir and diffs it against docs/api.

    Reports changed, missing (committed but no longer generated) and extra
    (generated but not committed) pages; empty when the trees match.
    """

    def _tree(self, root: Path, files: dict[str, str]) -> Path:
        for rel, text in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return root

    def test_identical_trees_have_no_drift(self, build_mod, tmp_path):
        files = {"a.md": "A\n", "data/b.md": "B\n"}
        gen = self._tree(tmp_path / "gen", files)
        committed = self._tree(tmp_path / "committed", files)
        assert build_mod.diff_trees(gen, committed) == []

    def test_changed_missing_and_extra_reported(self, build_mod, tmp_path):
        gen = self._tree(tmp_path / "gen", {"a.md": "A2\n", "data/new.md": "N\n"})
        committed = self._tree(
            tmp_path / "committed", {"a.md": "A\n", "data/stale.md": "S\n"}
        )
        drift = build_mod.diff_trees(gen, committed)
        assert sorted(drift) == [
            "changed: a.md",
            "missing (stale, delete): data/stale.md",
            "new (not committed): data/new.md",
        ]


class TestModulesRegistry:
    """MODULES and the myst.yml TOC list exactly the same API pages."""

    @pytest.fixture(scope="class")
    def toc_api_files(self):
        toc = yaml.safe_load((_ROOT / "docs" / "myst.yml").read_text())["project"][
            "toc"
        ]
        found: list[str] = []

        def walk(entries):
            for e in entries:
                if "file" in e and e["file"].startswith("api/"):
                    found.append(e["file"][len("api/") :])
                walk(e.get("children", []))

        walk(toc)
        return found

    def test_every_module_page_is_in_toc(self, build_mod, toc_api_files):
        assert sorted(out for _, out in build_mod.MODULES) == sorted(toc_api_files)

    def test_toc_has_no_duplicates(self, toc_api_files):
        assert len(toc_api_files) == len(set(toc_api_files))

    def test_stub_and_missing_pages(self, build_mod):
        modules = dict(build_mod.MODULES)
        assert "nltools.algorithms.inference.utils" not in modules
        assert modules["nltools.data.fitresults"] == "data/fitresults.md"
        assert modules["nltools.data.adjacency.spatial"] == "data/adjacency_spatial.md"
