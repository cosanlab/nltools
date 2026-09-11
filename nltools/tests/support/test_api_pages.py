"""Tests for scripts/check_api_pages.py and the docs/api page set.

The API reference is hand-written: each page under docs/api carries frontmatter,
prose, and `::: dotted.path` directives that mkdocstrings renders at build time.
These tests cover the checker's parsing and the whole-tree invariants — the
zensical nav and the page files agreeing, and every public export having exactly
one user-facing home.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).parents[3]
_SCRIPT = _ROOT / "scripts" / "check_api_pages.py"
_DOCS_API = _ROOT / "docs" / "api"


@pytest.fixture(scope="module")
def check_mod():
    """scripts/check_api_pages.py, loaded by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("check_api_pages", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestParseDirectives:
    """`::: identifier` blocks and the four-space `options:` block under them."""

    def test_bare_directive_has_no_members(self, check_mod):
        (directive,) = check_mod.parse_directives("# Page\n\n::: nltools.models\n")
        assert directive.identifier == "nltools.models"
        assert directive.members is None

    def test_members_list_is_read_in_order(self, check_mod):
        text = (
            "::: nltools.mask\n"
            "    options:\n"
            "      heading_level: 3\n"
            "      members:\n"
            "        - create_sphere\n"
            "        - expand_mask\n"
        )
        (directive,) = check_mod.parse_directives(text)
        assert directive.members == ["create_sphere", "expand_mask"]

    def test_members_false_documents_nothing(self, check_mod):
        text = "::: nltools.algorithms\n    options:\n      members: false\n"
        (directive,) = check_mod.parse_directives(text)
        assert directive.members is False
        assert check_mod.documented_objects([directive]) == {}

    def test_several_directives_on_one_page(self, check_mod):
        text = (
            "## A\n\n::: nltools.io\n    options:\n      members:\n        - to_h5\n"
            "\n## B\n\n::: nltools.datasets\n    options:\n      members:\n"
            "        - fetch_pain\n"
        )
        assert [d.identifier for d in check_mod.parse_directives(text)] == [
            "nltools.io",
            "nltools.datasets",
        ]

    def test_prose_colons_are_not_directives(self, check_mod):
        assert (
            check_mod.parse_directives("Nested ::: inline, and\n:::: not one\n") == []
        )

    def test_spaceless_and_indented_directives_are_seen(self, check_mod):
        """mkdocstrings accepts both, so the checker must not miss either."""
        text = (
            ":::nltools.models\n"
            "\n"
            "  ::: nltools.mask\n"
            "      options:\n"
            "        members:\n"
            "          - create_sphere\n"
        )
        directives = check_mod.parse_directives(text)
        assert [d.identifier for d in directives] == ["nltools.models", "nltools.mask"]
        assert directives[1].members == ["create_sphere"]

    def test_prose_after_a_blank_line_ends_the_options_block(self, check_mod):
        text = (
            "::: nltools.mask\n"
            "    options:\n"
            "      members:\n"
            "        - create_sphere\n"
            "\n"
            "Prose, then an indented block that is not options:\n"
            "\n"
            "    members:\n"
            "      - expand_mask\n"
        )
        (directive,) = check_mod.parse_directives(text)
        assert directive.members == ["create_sphere"]


class TestResolve:
    def test_module_and_attribute(self, check_mod):
        assert check_mod.resolve("nltools.mask").__name__ == "nltools.mask"
        assert check_mod.resolve("nltools.models.Ridge").__name__ == "Ridge"

    def test_missing_object_raises(self, check_mod):
        with pytest.raises(LookupError):
            check_mod.resolve("nltools.mask.no_such_function")


class TestObjectKey:
    """Identity for the exactly-once count: where an object is defined."""

    def test_an_alias_and_its_definition_are_one_object(self, check_mod):
        import nltools.algorithms
        import nltools.algorithms.corrections

        assert check_mod.object_key(nltools.algorithms.fdr) == check_mod.object_key(
            nltools.algorithms.corrections.fdr
        )

    def test_two_functions_are_two_objects(self, check_mod):
        import nltools.algorithms

        assert check_mod.object_key(nltools.algorithms.fdr) != check_mod.object_key(
            nltools.algorithms.holm_bonf
        )

    def test_values_without_a_definition_fall_back_to_identity(self, check_mod):
        sentinel = object()
        assert check_mod.object_key(sentinel) == id(sentinel)


class TestNav:
    """The zensical nav and the files under docs/api list the same pages."""

    def test_nav_and_files_agree(self, check_mod):
        nav = check_mod.api_nav()
        files = {p.relative_to(_DOCS_API).as_posix() for p in _DOCS_API.rglob("*.md")}
        assert set(nav) == files

    def test_internal_pages_are_grouped_under_internal_modules(self, check_mod):
        nav = check_mod.api_nav()
        internal = {
            rel for rel, groups in nav.items() if check_mod.INTERNAL_NAV_GROUP in groups
        }
        assert "utils.md" in internal
        assert "data/braindata_io.md" in internal
        assert "data/brain_data.md" not in internal
        assert not any(rel.startswith("tasks/") for rel in internal)

    def test_task_pages_are_the_eleven_task_pages(self, check_mod):
        nav = check_mod.api_nav()
        assert sorted(rel for rel in nav if rel.startswith("tasks/")) == sorted(
            f"tasks/{stem}.md"
            for stem in (
                "alignment",
                "atlases",
                "design-and-glm",
                "inference",
                "intersubject",
                "loading",
                "plotting",
                "prediction",
                "preprocessing",
                "similarity",
                "simulation",
            )
        )


class TestPages:
    """What each kind of page must contain."""

    def test_every_page_has_frontmatter_and_a_directive(self, check_mod):
        for path in sorted(_DOCS_API.rglob("*.md")):
            text = path.read_text()
            assert text.startswith("---\ntitle: "), path
            assert check_mod.parse_directives(text), path

    def test_task_pages_have_an_intro_and_explicit_members(self, check_mod):
        for path in sorted((_DOCS_API / "tasks").glob("*.md")):
            text = path.read_text()
            body = text.split("---\n", 2)[2]
            assert body.strip().startswith("# "), path
            prose = body.split("\n\n")[1]
            assert prose.strip() and not prose.startswith(":::"), path
            for directive in check_mod.parse_directives(text):
                # A task page re-cuts modules, so a module directive names the
                # members it takes; a directive may also name one object outright.
                names_a_module = isinstance(
                    check_mod.resolve(directive.identifier), ModuleType
                )
                assert isinstance(directive.members, list) or not names_a_module, (
                    path,
                    directive.identifier,
                )

    def test_no_page_renders_a_retired_module_whole(self, check_mod):
        """Modules that were collapsed into facades keep no page of their own.

        A task page may still name such a module and take a member or two from
        it; what is gone is the page that rendered the module in full.
        """
        whole_modules = {
            d.identifier
            for path in _DOCS_API.rglob("*.md")
            for d in check_mod.parse_directives(path.read_text())
            if d.members is None
        }
        assert whole_modules.isdisjoint(
            {
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
            }
        )
        assert "nltools.data.results" in whole_modules
        assert "nltools.algorithms.inference" in whole_modules


class TestCheck:
    """The gate `uv run poe lint-api` runs."""

    def test_check_passes(self, check_mod, capsys):
        assert check_mod.check() == 0
        assert (
            "each documented on exactly one user-facing page" in capsys.readouterr().out
        )
