"""Tests for the tutorial page generator in scripts/marimo_to_zensical.py.

The script isn't a package member, so it's loaded by file path via importlib.
Focus: the fence rewriting rules, the page slug that becomes a markdown-exec
session, the GitHub/molab links, and the banner. The ``marimo export md``
subprocess is never invoked, so these run without marimo.
"""

import importlib.util
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "marimo_to_zensical.py"

GLM_REL = "docs/tutorials/workflows/01_glm.py"
GLM_BLOB = "https://github.com/cosanlab/nltools/blob/master/" + GLM_REL
GLM_MOLAB = "https://molab.marimo.io/github/cosanlab/nltools/blob/master/" + GLM_REL
MOLAB_SHIELD = "https://marimo.io/molab-shield.svg"
SESSION = 'exec="on" session="workflows-01-glm"'
HIDDEN = f'{SESSION} render="off"'


@pytest.fixture(scope="module")
def m2z():
    spec = importlib.util.spec_from_file_location("marimo_to_zensical", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLinkHelpers:
    """Pure URL builders from a repo-relative posix path."""

    def test_github_url_is_master_blob(self, m2z):
        assert m2z.github_url(GLM_REL) == GLM_BLOB

    def test_molab_url_follows_official_github_pattern(self, m2z):
        # https://molab.marimo.io/github/{owner}/{repo}/blob/{branch}/{path}
        assert m2z.molab_url(GLM_REL) == GLM_MOLAB

    def test_molab_badge_is_official_snippet(self, m2z):
        assert (
            m2z.molab_badge(GLM_REL)
            == f"[![Open in molab]({MOLAB_SHIELD})]({GLM_MOLAB})"
        )


class TestPageSlug:
    """One markdown-exec session per page, named after the notebook."""

    def test_slug_joins_the_group_and_the_stem(self, m2z):
        assert m2z.page_slug(GLM_REL) == "workflows-01-glm"

    def test_underscores_in_the_stem_become_hyphens(self, m2z):
        assert m2z.page_slug("docs/tutorials/basics/01_brain_data.py") == (
            "basics-01-brain-data"
        )

    def test_every_notebook_has_its_own_slug(self, m2z):
        notebooks = [
            path.relative_to(_REPO_ROOT).as_posix()
            for pattern in m2z.TUTORIAL_GLOBS
            for path in sorted(_REPO_ROOT.glob(pattern))
        ]
        slugs = [m2z.page_slug(rel) for rel in notebooks]
        assert len(set(slugs)) == len(notebooks) == 7


class TestFrontmatter:
    def test_names_the_notebook_and_the_way_to_regenerate(self, m2z):
        front = m2z.frontmatter(GLM_REL, "GLM Analysis")
        assert front.startswith("---\n")
        assert (
            f"# AUTO-GENERATED from {GLM_REL} by scripts/marimo_to_zensical.py" in front
        )
        assert "DO NOT EDIT" in front
        assert "uv run poe docs-generate" in front

    def test_title_comes_from_the_notebook_heading(self, m2z):
        body = "# GLM Analysis\n\nSome prose.\n"
        assert m2z.page_title(body) == "GLM Analysis"
        assert "title: GLM Analysis" in m2z.frontmatter(GLM_REL, m2z.page_title(body))

    def test_a_marimo_frontmatter_block_is_dropped(self, m2z):
        text = "---\ntitle: 01 Glm\nmarimo-version: 0.24.0\n---\n\n# GLM Analysis\n"
        assert m2z.strip_frontmatter(text) == "\n# GLM Analysis\n"


class TestCellRewriting:
    """`python {.marimo}` fences become markdown-exec fences."""

    def test_a_plain_cell_executes_and_shows_its_source(self, m2z):
        out = m2z.transform_cell("", "x = 1", "workflows-01-glm")
        assert out == f'```python {SESSION} source="above"\nx = 1\n```'

    def test_hide_code_shows_the_output_only(self, m2z):
        out = m2z.transform_cell(' hide_code="true"', "x = 1", "workflows-01-glm")
        assert out == f"```python {SESSION}\nx = 1\n```"

    def test_a_hidden_cell_runs_but_renders_nothing(self, m2z):
        # `render="off"` is what makes it render nothing; without it this is
        # the hide_code fence above, which shows the cell's output.
        out = m2z.transform_cell("", "# docs: hide\nx = 1", "workflows-01-glm")
        assert out == f"```python {HIDDEN}\nx = 1\n```"
        assert out != f"```python {SESSION}\nx = 1\n```"

    def test_marimo_imports_are_stripped(self, m2z):
        out = m2z.transform_cell("", "import marimo as mo\nx = 1", "workflows-01-glm")
        assert "marimo" not in out

    def test_a_cell_left_empty_is_dropped(self, m2z):
        assert m2z.transform_cell("", "import marimo as mo", "workflows-01-glm") is None

    def test_the_fence_grows_past_a_fence_in_the_cell(self, m2z):
        out = m2z.transform_cell("", 'doc = """\n```\n"""', "workflows-01-glm")
        assert out.startswith("````python ")
        assert out.endswith("\n````")

    def test_the_cell_regex_matches_a_lengthened_marimo_fence(self, m2z):
        exported = '````python {.marimo}\ndoc = """\n```\n"""\n````'
        match = m2z.CELL_RE.search(exported)
        assert match.group("body") == 'doc = """\n```\n"""'


class TestInstallCell:
    """Every page opens with a hidden cell that installs the formatter."""

    def test_the_install_cell_is_hidden_and_names_the_notebook(self, m2z):
        out = m2z.install_cell(GLM_REL, "workflows-01-glm")
        assert out == (
            f"```python {HIDDEN}\nimport docs_show\n\n"
            f'docs_show.install("{GLM_REL}")\n```'
        )


class TestBanner:
    def test_the_banner_is_a_badge_and_a_tip_admonition(self, m2z):
        banner = m2z.source_banner(GLM_REL)
        assert banner.startswith(m2z.molab_badge(GLM_REL))
        assert '!!! tip "Run this tutorial"' in banner
        assert "    This page is rendered from the [marimo]" in banner
        assert "uvx marimo edit --sandbox 01_glm.py" in banner

    def test_the_banner_lands_under_the_top_level_heading(self, m2z):
        body = "# GLM Analysis\n\nSome prose.\n"
        assert m2z.insert_after_heading(body, "BANNER") == (
            "# GLM Analysis\n\nBANNER\nSome prose.\n"
        )

    def test_a_comment_inside_a_code_fence_is_not_a_heading(self, m2z):
        body = "```python\n# not a heading\n```\n\n# GLM Analysis\n"
        assert m2z.insert_after_heading(body, "BANNER").endswith(
            "# GLM Analysis\n\nBANNER\n"
        )


class TestBlankLines:
    """Blank-line runs are prose formatting; a cell's own are its source."""

    def test_a_run_of_blank_lines_in_prose_collapses(self, m2z):
        assert m2z.collapse_blank_lines("one\n\n\n\ntwo") == "one\n\ntwo"

    def test_blank_lines_inside_a_cell_survive(self, m2z):
        text = "```python\nimport os\n\n\ndef fit():\n    pass\n```"
        assert m2z.collapse_blank_lines(text) == text

    def test_a_lengthened_fence_closes_on_its_own_guard(self, m2z):
        text = '````python\na = """\n```\n\n\n"""\n````\n\n\nprose'
        assert m2z.collapse_blank_lines(text) == (
            '````python\na = """\n```\n\n\n"""\n````\n\nprose'
        )


class TestAdmonitions:
    """marimo `///` fences become `!!!` admonitions with an indented body."""

    def test_a_titled_fence_keeps_its_title(self, m2z):
        text = "/// tip | Try this\nBody text.\n///\n"
        assert m2z.convert_admonitions(text) == '!!! tip "Try this"\n    Body text.\n'

    def test_an_untitled_fence_has_no_title(self, m2z):
        text = "/// note\nBody text.\n///\n"
        assert m2z.convert_admonitions(text) == "!!! note\n    Body text.\n"

    def test_prose_is_left_alone(self, m2z):
        text = "Some prose.\n\nMore prose.\n"
        assert m2z.convert_admonitions(text) == text


def test_generated_pages_are_the_nav_entries():
    """Guide links and the nav point at `tutorials/<group>/<stem>.md`."""
    nav = tomllib.loads((_REPO_ROOT / "zensical.toml").read_text())["project"]["nav"]
    tutorials = next(entry["Tutorials"] for entry in nav if "Tutorials" in entry)
    pages = {
        page
        for group in tutorials
        for pages in group.values()
        if isinstance(pages, list)
        for page in pages
    }
    expected = {
        f"tutorials/{path.parent.name}/{path.stem}.md"
        for pattern in [
            "docs/tutorials/basics/[0-9]*.py",
            "docs/tutorials/workflows/[0-9]*.py",
        ]
        for path in _REPO_ROOT.glob(pattern)
    }
    assert pages == expected
