"""Tests for the MyST -> Zensical page converter in scripts/myst_to_zensical.py.

The script isn't a package member, so it's loaded by file path via importlib.
One test per conversion rule the converter promises.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "myst_to_zensical.py"


@pytest.fixture(scope="module")
def m2z():
    spec = importlib.util.spec_from_file_location("myst_to_zensical", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # `dataclass` resolves annotations through sys.modules, so register first.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestFrontmatter:
    """Zensical reads `title` and `description`; everything else is MyST-only."""

    def test_keeps_title_and_description(self, m2z):
        page = "---\ntitle: Benchmarks\ndescription: Speed tables.\n---\n\nBody.\n"
        assert m2z.convert(page).text == page

    def test_drops_myst_only_keys(self, m2z):
        page = (
            "---\n"
            "title: BrainData Basics\n"
            "kernelspec:\n"
            "  name: python3\n"
            "downloads:\n"
            "  - url: https://example.org/nb.py\n"
            "    title: Source notebook\n"
            "---\n"
            "\n"
            "Body.\n"
        )
        assert m2z.convert(page).text == "---\ntitle: BrainData Basics\n---\n\nBody.\n"

    def test_drops_a_frontmatter_block_with_nothing_left(self, m2z):
        page = "---\nlabel: page-tasks-atlases\n---\n\nBody.\n"
        assert m2z.convert(page).text == "Body.\n"

    def test_leaves_a_page_without_frontmatter_alone(self, m2z):
        page = "# Title\n\nBody.\n"
        assert m2z.convert(page).text == page


class TestAdmonitions:
    """`:::{name}` directives become `!!!` blocks with an indented body."""

    def test_untitled_admonition(self, m2z):
        page = ":::{note}\nWatch out.\n:::\n"
        assert m2z.convert(page).text == "!!! note\n    Watch out.\n"

    def test_titled_admonition(self, m2z):
        page = ":::{note} Auto-generated\nDo not edit by hand.\n:::\n"
        assert (
            m2z.convert(page).text
            == '!!! note "Auto-generated"\n    Do not edit by hand.\n'
        )

    @pytest.mark.parametrize("name", ["note", "tip", "warning"])
    def test_each_admonition_name_is_kept(self, m2z, name):
        assert m2z.convert(f":::{{{name}}}\nText.\n:::\n").text.startswith(
            f"!!! {name}"
        )

    def test_body_paragraphs_and_fences_are_indented(self, m2z):
        page = ":::{tip}\nRun it:\n\n```python\nx = 1\n```\n:::\n"
        assert m2z.convert(page).text == (
            "!!! tip\n    Run it:\n\n    ```python\n    x = 1\n    ```\n"
        )

    def test_dropdown_becomes_a_details_block(self, m2z):
        page = ":::{dropdown} Show the derivation\nIt follows.\n:::\n"
        assert (
            m2z.convert(page).text
            == '??? note "Show the derivation"\n    It follows.\n'
        )


class TestLabelsAndCrossReferences:
    """MyST `(label)=` targets become heading anchors that `attr_list` renders."""

    def test_target_above_a_heading_becomes_a_heading_anchor(self, m2z):
        page = "(renamed-kwargs)=\n## Renamed kwargs\n\nBody.\n"
        assert (
            m2z.convert(page).text == "## Renamed kwargs {#renamed-kwargs}\n\nBody.\n"
        )

    def test_blank_line_between_target_and_heading_is_allowed(self, m2z):
        page = "(renamed-kwargs)=\n\n## Renamed kwargs\n"
        assert m2z.convert(page).text == "## Renamed kwargs {#renamed-kwargs}\n"

    def test_target_that_labels_a_paragraph_is_dropped(self, m2z):
        page = "(orphan-target)=\nJust a paragraph.\n"
        assert m2z.convert(page).text == "Just a paragraph.\n"

    def test_empty_cross_reference_gets_the_heading_text(self, m2z):
        page = "(tail-vocabulary)=\n## Tail vocabulary\n\nSee [](#tail-vocabulary).\n"
        assert "See [Tail vocabulary](#tail-vocabulary)." in m2z.convert(page).text

    def test_cross_reference_with_its_own_text_is_untouched(self, m2z):
        page = "(tail-vocabulary)=\n## Tail vocabulary\n\nSee [the tails](#tail-vocabulary).\n"
        assert "See [the tails](#tail-vocabulary)." in m2z.convert(page).text

    def test_unresolvable_cross_reference_is_reported(self, m2z):
        result = m2z.convert("See [](#nowhere).\n")
        assert result.text == "See [](#nowhere).\n"
        assert result.unconverted == ("[](#nowhere)",)


class TestTablePipes:
    """No markdown escape survives both engines, so such spans become raw `<code>`."""

    def test_code_span_with_an_escaped_pipe_becomes_raw_code(self, m2z):
        page = "| Kwarg |\n|---|\n| `summary: 'mean' \\| 'median'` |\n"
        assert (
            "| <code>summary: 'mean' &#124; 'median'</code> |" in m2z.convert(page).text
        )

    def test_angle_brackets_and_ampersands_in_the_span_are_escaped(self, m2z):
        page = "| Kwarg |\n|---|\n| `x: int \\| None <T> & U` |\n"
        assert (
            "<code>x: int &#124; None &lt;T&gt; &amp; U</code>"
            in m2z.convert(page).text
        )

    def test_code_span_without_a_pipe_stays_markdown(self, m2z):
        page = "| Kwarg |\n|---|\n| `n_jobs: int = -1` |\n"
        assert m2z.convert(page).text == page

    def test_escaped_pipe_outside_a_code_span_stays_escaped(self, m2z):
        page = "| Choice |\n|---|\n| mean \\| median |\n"
        assert m2z.convert(page).text == page

    def test_a_table_without_leading_pipes_is_still_a_table(self, m2z):
        page = "Kwarg | Notes\n--- | ---\n`x: 'a' \\| 'b'` | pick one\n"
        assert "<code>x: 'a' &#124; 'b'</code> | pick one" in m2z.convert(page).text

    def test_a_line_that_is_not_a_table_row_is_left_alone(self, m2z):
        page = "Pass `summary='mean' \\| 'median'` to choose.\n"
        assert m2z.convert(page).text == page

    def test_a_fenced_code_block_is_left_alone(self, m2z):
        page = "```text\n| A |\n|---|\n| `a \\| b` |\n```\n"
        assert m2z.convert(page).text == page


class TestFencesLeftAlone:
    """Bibliographies go; executable cells are the tutorial pipeline's business."""

    def test_bibliography_block_is_dropped(self, m2z):
        page = (
            "Text.\n\n```{bibliography}\n:filter: docname in docnames\n```\n\nMore.\n"
        )
        assert m2z.convert(page).text == "Text.\n\nMore.\n"

    def test_code_cell_is_untouched(self, m2z):
        page = "```{code-cell} python3\nfrom nltools import BrainData\n```\n"
        assert m2z.convert(page).text == page


class TestUnknownDirectives:
    """What the converter cannot express is left in place and named."""

    def test_unknown_directive_is_kept_and_reported(self, m2z):
        page = "::::{grid} 1 2 2 2\n\n:::{grid-item-card} Basics\nText.\n:::\n\n::::\n"
        result = m2z.convert(page)
        assert "::::{grid} 1 2 2 2" in result.text
        assert result.unconverted == ("::::{grid}", ":::{grid-item-card}")
