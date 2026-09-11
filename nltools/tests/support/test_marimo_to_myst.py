"""Tests for the tutorial converter in scripts/marimo_to_myst.py.

The script isn't a package member, so it's loaded by file path via importlib.
Focus: the GitHub/molab link helpers, the per-page frontmatter (edit/source
links + downloads), the "Open in molab" banner, and banner placement. The
``marimo export md`` subprocess is stubbed so these run without marimo.
"""

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "marimo_to_myst.py"

GLM_REL = "docs/tutorials/workflows/01_glm.py"
GLM_BLOB = "https://github.com/cosanlab/nltools/blob/master/" + GLM_REL
GLM_EDIT = "https://github.com/cosanlab/nltools/edit/master/" + GLM_REL
GLM_MOLAB = "https://molab.marimo.io/github/cosanlab/nltools/blob/master/" + GLM_REL
MOLAB_SHIELD = "https://marimo.io/molab-shield.svg"


@pytest.fixture(scope="module")
def m2m():
    spec = importlib.util.spec_from_file_location("marimo_to_myst", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLinkHelpers:
    """Pure URL builders from a repo-relative posix path."""

    def test_github_url_is_master_blob(self, m2m):
        assert m2m.github_url(GLM_REL) == GLM_BLOB

    def test_github_edit_url(self, m2m):
        assert m2m.github_edit_url(GLM_REL) == GLM_EDIT

    def test_molab_url_follows_official_github_pattern(self, m2m):
        # https://molab.marimo.io/github/{owner}/{repo}/blob/{branch}/{path}
        assert m2m.molab_url(GLM_REL) == GLM_MOLAB

    def test_molab_badge_is_official_snippet(self, m2m):
        assert (
            m2m.molab_badge(GLM_REL)
            == f"[![Open in molab]({MOLAB_SHIELD})]({GLM_MOLAB})"
        )


class TestFrontmatter:
    def test_keeps_autogen_comment_with_repo_relative_source(self, m2m):
        fm = m2m.frontmatter(GLM_REL)
        assert fm.startswith("---\n")
        assert f"# AUTO-GENERATED from {GLM_REL} by scripts/marimo_to_myst.py" in fm
        assert "DO NOT EDIT" in fm
        assert "uv run poe docs-generate" in fm

    def test_kernelspec_kept(self, m2m):
        fm = m2m.frontmatter(GLM_REL)
        assert "kernelspec:\n  name: python3\n  display_name: Python 3\n" in fm

    def test_edit_and_source_urls_point_at_the_notebook(self, m2m):
        fm = m2m.frontmatter(GLM_REL)
        assert f"edit_url: {GLM_EDIT}\n" in fm
        assert f"source_url: {GLM_BLOB}\n" in fm

    def test_downloads_has_molab_and_source_entries(self, m2m):
        fm = m2m.frontmatter(GLM_REL)
        assert "downloads:\n" in fm
        assert f"  - url: {GLM_MOLAB}\n    title: Open in molab\n" in fm
        assert f"  - url: {GLM_BLOB}\n    title: Source notebook (01_glm.py)\n" in fm

    def test_ends_with_closing_delimiter(self, m2m):
        assert m2m.frontmatter(GLM_REL).endswith("\n---\n")


class TestSourceBanner:
    def test_badge_paragraph_precedes_tip(self, m2m):
        banner = m2m.source_banner(GLM_REL)
        badge, rest = banner.split("\n\n", 1)
        assert badge == m2m.molab_badge(GLM_REL)
        assert rest.startswith("````{tip}")
        assert rest.rstrip("\n").endswith("````")

    def test_tip_has_local_and_cloud_instructions(self, m2m):
        banner = m2m.source_banner(GLM_REL)
        assert f"[`{GLM_REL}`]({GLM_BLOB})" in banner
        assert "uvx marimo edit --sandbox 01_glm.py" in banner
        assert "baked in at build time" in banner
        assert "no install" in banner


class TestInsertBanner:
    BANNER = "BADGE\n\n````{tip} x\nbody\n````\n"

    def test_badge_directly_after_h1(self, m2m):
        body = "# Title\n\nIntro paragraph.\n"
        out = m2m.insert_banner(body, self.BANNER)
        assert out.startswith(
            "# Title\n\nBADGE\n\n````{tip} x\nbody\n````\n\nIntro paragraph."
        )

    def test_ignores_hash_lines_inside_code_fences(self, m2m):
        body = (
            "```python {.marimo}\n"
            "# not a heading\n"
            "x = 1\n"
            "```\n"
            "\n"
            "# Real Title\n"
            "\n"
            "Intro.\n"
        )
        out = m2m.insert_banner(body, self.BANNER)
        assert "# not a heading\nx = 1\n```" in out  # fence untouched
        assert out.index("BADGE") > out.index("# Real Title")
        assert "# Real Title\n\nBADGE\n" in out

    def test_prepends_when_no_heading(self, m2m):
        out = m2m.insert_banner("just text\n", self.BANNER)
        assert out.startswith("BADGE\n")


class TestConvert:
    """End-to-end on a stubbed `marimo export md` result, inside the repo tree."""

    def test_output_has_frontmatter_badge_and_code_cell(
        self, m2m, monkeypatch, tmp_path
    ):
        exported = (
            "---\ntitle: T\nmarimo-version: 0.23.10\n---\n\n"
            "# Title\n\n"
            "Prose.\n\n"
            "```python {.marimo}\nimport marimo as mo\n```\n\n"
            '```python {.marimo hide_code="true"}\nprint(1)\n```\n'
        )
        monkeypatch.setattr(m2m, "export_marimo_md", lambda nb: exported)
        # Treat tmp_path as the repo root so rel_to_repo yields a repo-relative path.
        root = tmp_path.resolve()
        monkeypatch.setattr(m2m, "REPO_ROOT", root)
        rel = "docs/tutorials/workflows/99_fake.py"
        nb = root / rel
        nb.parent.mkdir(parents=True)
        nb.write_text("# placeholder\n")
        out = m2m.convert(nb)
        assert out == nb.with_suffix(".md")
        text = out.read_text()
        assert text.startswith(m2m.frontmatter(rel))
        assert f"# Title\n\n{m2m.molab_badge(rel)}\n\n````{{tip}}" in text
        assert "import marimo" not in text
        assert "```{code-cell} python3\n:tags: [remove-input]\nprint(1)\n```" in text


def _committed_notebooks() -> list[Path]:
    return sorted(
        p
        for pattern in (
            "docs/tutorials/basics/[0-9]*.py",
            "docs/tutorials/workflows/[0-9]*.py",
        )
        for p in _REPO_ROOT.glob(pattern)
    )


@pytest.mark.parametrize("notebook", _committed_notebooks(), ids=lambda p: p.stem)
def test_committed_md_is_in_sync_with_notebook(m2m, notebook):
    """Each committed tutorial .md carries the frontmatter and molab badge of its .py.

    Regenerate with `uv run poe docs-generate` when this fails.
    """
    md = notebook.with_suffix(".md")
    assert md.exists(), f"{md} missing — run `uv run poe docs-generate`"
    rel = notebook.relative_to(_REPO_ROOT).as_posix()
    text = md.read_text()
    hint = f"{md.name} is stale — run `uv run poe docs-generate`"
    assert text.startswith(m2m.frontmatter(rel)), hint
    assert m2m.molab_badge(rel) in text, hint
