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


def _toc_api_files() -> list[str]:
    toc = yaml.safe_load((_ROOT / "docs" / "myst.yml").read_text())["project"]["toc"]
    found: list[str] = []

    def walk(entries):
        for e in entries:
            if "file" in e and e["file"].startswith("api/"):
                found.append(e["file"][len("api/") :])
            walk(e.get("children", []))

    walk(toc)
    return found


class TestPagesRegistry:
    """PAGES and the myst.yml TOC list exactly the same API pages.

    Every generated page appears in the TOC exactly once, and every ``api/``
    TOC entry is generated. Pages are either module pages (one griffe object
    rendered with griffe2md's template), a namespace page (every public name
    of a package, A-Z), or task pages (a hand-written intro over an explicit
    object list).
    """

    def test_every_generated_page_is_in_toc_exactly_once(self, build_mod):
        toc = _toc_api_files()
        assert sorted(p.output for p in build_mod.PAGES) == sorted(toc)
        assert len(toc) == len(set(toc))

    def test_each_page_has_exactly_one_source(self, build_mod):
        for page in build_mod.PAGES:
            sources = [page.module, page.namespace, page.objects or None]
            assert sum(s is not None for s in sources) == 1, page.output

    def test_task_pages_have_intro_and_objects(self, build_mod):
        tasks = [p for p in build_mod.PAGES if p.output.startswith("tasks/")]
        assert sorted(p.output for p in tasks) == sorted(
            f"tasks/{stem}.md"
            for stem in (
                "loading",
                "preprocessing",
                "design-and-glm",
                "prediction",
                "similarity",
                "alignment",
                "inference",
                "intersubject",
                "plotting",
                "atlases",
                "simulation",
            )
        )
        for page in tasks:
            assert page.intro.strip(), page.output
            assert page.objects, page.output
            assert not page.internal, page.output

    def test_deleted_module_pages_are_gone(self, build_mod):
        modules = {p.module for p in build_mod.PAGES if p.module}
        for gone in (
            "nltools.plotting",
            "nltools.mask",
            "nltools.io",
            "nltools.datasets",
            "nltools.cross_validation",
            "nltools.data.roc",
            "nltools.data.simulator",
            "nltools.data.atlases",
            "nltools.algorithms.corrections",
            "nltools.algorithms.inference.one_sample",
            "nltools.algorithms.inference.utils",
        ):
            assert gone not in modules
        by_module = {p.module: p.output for p in build_mod.PAGES if p.module}
        assert by_module["nltools.data.results"] == "data/results.md"
        assert by_module["nltools.algorithms.inference"] == "algorithms/inference.md"

    def test_a_to_z_index_is_the_algorithms_namespace(self, build_mod):
        (index,) = [p for p in build_mod.PAGES if p.output == "algorithms.md"]
        assert index.namespace == "nltools.algorithms"
        assert not index.internal


class TestComposeObjectsPage:
    """Task pages: intro, per-kind summary tables, then category sections.

    Attributes appear in their summary table only (no detail section — the
    same policy the postprocess applies to module pages). Members keep the
    registry order within a category.
    """

    def _members(self, build_mod):
        M = build_mod.Member
        return [
            M("fdr", "function", "FDR threshold.", "### `fdr`\n\nBody f.\n"),
            M("Roc", "class", "ROC analysis.", "### `Roc`\n\nBody r.\n"),
            M("ATLASES", "attribute", "", "", annotation="dict[str, AtlasMetadata]"),
            M("zscore", "function", "Z-score.", "### `zscore`\n\nBody z.\n"),
        ]

    def test_layout(self, build_mod):
        out = build_mod.compose_objects_page("Intro line.", self._members(build_mod))
        assert out.startswith("Intro line.\n\n")
        # Summary tables in canonical order, then Classes before Functions.
        order = [
            "**Attributes:**",
            "`ATLASES` | <code>dict[str, AtlasMetadata]</code> |",
            "**Classes:**",
            "[`Roc`](#roc) | ROC analysis.",
            "**Functions:**",
            "[`fdr`](#fdr) | FDR threshold.",
            "[`zscore`](#zscore) | Z-score.",
            "## Classes",
            "### `Roc`",
            "## Functions",
            "### `fdr`",
            "### `zscore`",
        ]
        positions = [out.index(s) for s in order]
        assert positions == sorted(positions), out
        assert "## Attributes" not in out

    def test_missing_kinds_are_omitted(self, build_mod):
        M = build_mod.Member
        out = build_mod.compose_objects_page(
            "I.", [M("f", "function", "F.", "### `f`\n\nB.\n")]
        )
        assert "Classes" not in out and "Attributes" not in out


# Public namespaces whose every ``__all__`` name must be documented on a page
# a user is expected to browse (a task page, a class page, or the A-Z index).
PUBLIC_NAMESPACES = [
    "nltools",
    "nltools.data",
    "nltools.algorithms",
    "nltools.plotting",
    "nltools.mask",
    "nltools.io",
    "nltools.datasets",
    "nltools.cross_validation",
    "nltools.templates",
    "nltools.models",
    "nltools.data.atlases",
    "nltools.utils",
]

# Exported names whose home is an internal-module page by design: low-level
# helpers with no user-facing task (they are still checked to be documented
# somewhere, just not required on a task/class/index page).
INTERNAL_ONLY = {
    # nltools.utils plumbing (progress bars, reserved column names, imports).
    "RESERVED_PREFIX",
    "all_same",
    "attempt_to_import",
    "coalesced_gc",
    "find_stack_level",
    "get_resource_path",
    "is_reserved_name",
    "make_progress_bar",
    "maybe_tqdm",
    "parse_run_separated",
    "reserved_name",
    "run_separated_name",
    # nltools.templates resolution internals behind get_brainspace/set_brainspace.
    "TemplateMatch",
    "match_resolution",
    "resolve_paths",
    "resolve_template_name",
}

# Exported module constants with no docstring: griffe2md hides them
# (``show_if_no_docstring = false``), so no page can carry them until the
# constant gets a docstring at its definition.
UNDOCUMENTED_CONSTANTS = set()


@pytest.fixture(scope="module")
def site(build_mod, tmp_path_factory):
    """Generate the whole docs/api tree once (griffe loads nltools in-process)."""
    out = tmp_path_factory.mktemp("api")
    report = build_mod.build(out, verbose=False)
    assert not report.failed, report.failed
    return report


class TestGeneratedSite:
    """Whole-tree invariants of the generated reference.

    Runs the real pipeline into a temp dir: explicit labels are unique across
    every page (mystmd otherwise reports duplicate identifiers), every public
    name is documented where a user will find it, and the A-Z index carries
    ``procrustes`` (whose name griffe resolves to the shadowing submodule).
    """

    def test_labels_unique_across_all_pages(self, build_mod, site):
        import postprocess_api_docs as pp

        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for rel, text in site.pages.items():
            for label in pp.collect_labels(text):
                if label in seen:
                    duplicates.append(f"{label}: {seen[label]} and {rel}")
                seen[label] = rel
        assert duplicates == []

    def test_procrustes_is_on_the_a_to_z_index(self, site):
        import postprocess_api_docs as pp

        assert "procrustes" in pp.documented_names(site.pages["algorithms.md"])

    @pytest.mark.parametrize("namespace", PUBLIC_NAMESPACES)
    def test_public_names_documented_on_a_user_facing_page(
        self, build_mod, site, namespace
    ):
        import importlib
        import types

        import postprocess_api_docs as pp

        pages_by_output = {p.output: p for p in build_mod.PAGES}
        user_facing: set[str] = set()
        anywhere: set[str] = set()
        for rel, text in site.pages.items():
            page = pages_by_output[rel]
            names = pp.documented_names(text)
            if page.module:
                # A module/class page documents its own object (the root
                # heading is stripped in favour of the frontmatter title).
                names.add(page.module.rsplit(".", 1)[-1])
            anywhere |= names
            if not page.internal:
                user_facing |= names

        mod = importlib.import_module(namespace)
        missing: list[str] = []
        for name in mod.__all__:
            if (
                name.startswith("__")
                or name in UNDOCUMENTED_CONSTANTS
                or isinstance(getattr(mod, name), types.ModuleType)
            ):
                continue
            required_in = anywhere if name in INTERNAL_ONLY else user_facing
            if name not in required_in:
                missing.append(name)
        assert missing == [], f"{namespace}: not documented: {missing}"
