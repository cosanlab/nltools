"""Tests for the API-doc postprocess helpers in scripts/build_api_docs.py.

The script isn't a package member, so it's loaded by file path via importlib.
Focus: the postprocess safety nets that scrub third-party RST leakage from
griffe2md output (nltools' own docstrings are Markdown-only by policy).
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "build_api_docs.py"


@pytest.fixture(scope="module")
def build_api_docs():
    spec = importlib.util.spec_from_file_location("build_api_docs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStripRstDirectives:
    """`.. directive::` blocks (e.g. nilearn_deprecated) must not leak in.

    These come from re-exported third-party functions (nltools.algorithms.hrf
    re-exports nilearn's glover_hrf/spm_hrf/etc), whose docstrings are RST. We
    can't rewrite them at the source, so postprocess strips the block.
    """

    def test_strips_nilearn_deprecated_block(self, build_api_docs):
        # Mirrors the real leak in docs/api/algorithms.md: an indented directive
        # marker plus a further-indented body, inside a param description.
        text = (
            "tr:\n"
            "\n"
            "    .. nilearn_deprecated:: 0.11.0\n"
            "\n"
            "        Use ``t_r`` instead (see above).\n"
            "\n"
            '<details class="time_length-" open markdown="1">\n'
        )
        out = build_api_docs._strip_rst_directives(text)
        assert "nilearn_deprecated" not in out
        assert "Use ``t_r`` instead" not in out
        # Surrounding content survives.
        assert "tr:" in out
        assert '<details class="time_length-" open markdown="1">' in out

    def test_strips_bodyless_directive(self, build_api_docs):
        text = "before\n\n.. versionadded:: 0.9\n\nafter\n"
        out = build_api_docs._strip_rst_directives(text)
        assert "versionadded" not in out
        assert "before" in out and "after" in out

    def test_leaves_non_directive_prose_untouched(self, build_api_docs):
        # A lone ".." or a role is not a block directive; don't touch it here.
        text = "See the note below.\n\n`t_r` is the repetition time.\n"
        out = build_api_docs._strip_rst_directives(text)
        assert out == text


class TestRemoveAttributesSections:
    """Attributes detail sections drop, but must not swallow class headings.

    Regression for the heading-depth bug: the old terminator ("next Methods
    anywhere") let a module-level `### Attributes` section greedily consume the
    following `### Classes` / `#### FirstClass` headings up to the first class's
    Methods, collapsing the page to a module-h2 -> class-Methods-h5 jump that
    mystmd warns on. Removal must be level-aware: an Attributes heading at level
    L ends at the next heading of level <= L (its sibling/parent).
    """

    def _page(self):
        # Mirrors real griffe2md structure after the concatenated-heading split:
        # module attrs (h3) -> Classes (h3) -> FirstClass (h4) with its own
        # attrs (h5) + methods (h5).
        return (
            "## `cv`\n\n"
            "### Attributes\n\n"
            "#### `CVSchemeType`\n\n"
            "A module-level type alias.\n\n"
            "### Classes\n\n"
            "#### `CVScheme`\n\n"
            "The class docstring.\n\n"
            "##### Attributes\n\n"
            "###### `k`\n\n"
            "Number of folds.\n\n"
            "##### Methods\n\n"
            "###### `split`\n\n"
            "Yield splits.\n"
        )

    def test_class_heading_survives(self, build_api_docs):
        out = build_api_docs._remove_attributes_sections(self._page())
        # The first class's detail heading must NOT be eaten.
        assert "#### `CVScheme`" in out
        assert "### Classes" in out
        # No heading-depth jump: h2 is followed by h3, never straight to h5.
        assert "## `cv`\n\n##### Methods" not in out

    def test_attributes_detail_removed(self, build_api_docs):
        out = build_api_docs._remove_attributes_sections(self._page())
        # Both the module-level and class-level Attributes detail go away...
        assert "### Attributes" not in out
        assert "##### Attributes" not in out
        assert "#### `CVSchemeType`" not in out
        assert "###### `k`" not in out
        # ...but Methods and its members stay.
        assert "##### Methods" in out
        assert "###### `split`" in out
