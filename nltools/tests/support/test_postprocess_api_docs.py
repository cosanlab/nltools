"""Tests for the API-doc postprocess helpers in scripts/postprocess_api_docs.py.

The script isn't a package member, so it's loaded by file path via importlib.
Focus: the postprocess passes that turn griffe2md output into mystmd-clean
pages (RST safety nets, summary-table ordering, heading/table cleanup, titles).
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "postprocess_api_docs.py"


@pytest.fixture(scope="module")
def postprocess_mod():
    spec = importlib.util.spec_from_file_location("postprocess_api_docs", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestStripRstDirectives:
    """`.. directive::` blocks (e.g. nilearn_deprecated) must not leak in.

    A safety net for directly re-exported third-party functions, whose RST
    docstrings can't be rewritten at the source (nltools currently has none;
    the hrf module wraps nilearn's HRFs behind its own docstrings).
    """

    def test_strips_nilearn_deprecated_block(self, postprocess_mod):
        # Mirrors the leak the old hrf re-exports produced: an indented directive
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
        out = postprocess_mod._strip_rst_directives(text)
        assert "nilearn_deprecated" not in out
        assert "Use ``t_r`` instead" not in out
        # Surrounding content survives.
        assert "tr:" in out
        assert '<details class="time_length-" open markdown="1">' in out

    def test_strips_bodyless_directive(self, postprocess_mod):
        text = "before\n\n.. versionadded:: 0.9\n\nafter\n"
        out = postprocess_mod._strip_rst_directives(text)
        assert "versionadded" not in out
        assert "before" in out and "after" in out

    def test_leaves_non_directive_prose_untouched(self, postprocess_mod):
        # A lone ".." or a role is not a block directive; don't touch it here.
        text = "See the note below.\n\n`t_r` is the repetition time.\n"
        out = postprocess_mod._strip_rst_directives(text)
        assert out == text


class TestReorderSummaryBlocks:
    """Summary tables reorder to Parameters -> Attributes -> Classes -> Methods.

    griffe emits the member summary tables Functions-before-Attributes (and
    Modules-before-Classes on module pages), a less natural reading order. The
    reorder pass rewrites each contiguous run of `**X:**` summary blocks to the
    canonical order, while leaving per-member Parameters/Returns/Examples runs
    untouched (Parameters is already first there, so it is a no-op).
    """

    def _class_page(self):
        # Mirrors docs/api/data/brain_data.md: a class heading followed by its
        # constructor Parameters, a Methods summary table, an Attributes summary
        # table, then a per-method detail section with its own runs.
        return (
            "## `BrainData`\n\n"
            "The class docstring.\n\n"
            "**Parameters:**\n\n"
            "| Name | Type |\n| --- | --- |\n| `data` | str |\n\n"
            "**Methods:**\n\n"
            "| Name | Description |\n| --- | --- |\n| [`align`](#align) | Align. |\n\n"
            "**Attributes:**\n\n"
            "| Name | Description |\n| --- | --- |\n| `shape` | Shape. |\n\n"
            "### Methods\n\n"
            "#### `align`\n\n"
            "Align the data.\n\n"
            "**Parameters:**\n\n"
            "| Name | Type |\n| --- | --- |\n| `target` | BrainData |\n\n"
            "**Returns:**\n\n"
            "BrainData\n\n"
            "**Examples:**\n\n"
            "```python\nbd.align(other)\n```\n"
        )

    def test_attributes_precede_methods_in_class_summary(self, postprocess_mod):
        out = postprocess_mod._reorder_summary_blocks(self._class_page())
        # In the class summary run, Attributes now comes before Methods.
        assert out.index("**Attributes:**") < out.index("**Methods:**")
        # The table rows travel with their block (no content scrambling).
        assert out.index("| `shape` | Shape. |") < out.index("[`align`](#align)")

    def test_per_method_run_is_untouched(self, postprocess_mod):
        out = postprocess_mod._reorder_summary_blocks(self._class_page())
        # Within the `align` detail, Parameters -> Returns -> Examples is kept:
        # the method's Parameters is the LAST **Parameters:** on the page.
        p = out.rfind("**Parameters:**")
        assert p < out.index("**Returns:**") < out.index("**Examples:**")

    def test_module_summary_puts_classes_before_modules(self, postprocess_mod):
        page = (
            "## `data`\n\n"
            "**Modules:**\n\n"
            "| Name | Description |\n| --- | --- |\n| `io` | IO. |\n\n"
            "**Classes:**\n\n"
            "| Name | Description |\n| --- | --- |\n| `BrainData` | Brain. |\n\n"
            "### Classes\n\n#### `BrainData`\n\nThe class.\n"
        )
        out = postprocess_mod._reorder_summary_blocks(page)
        assert out.index("**Classes:**") < out.index("**Modules:**")

    def test_single_block_run_is_noop(self, postprocess_mod):
        page = "## `x`\n\n**Parameters:**\n\n| a | b |\n| --- | --- |\n\n### Methods\n"
        assert postprocess_mod._reorder_summary_blocks(page) == page


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

    def test_class_heading_survives(self, postprocess_mod):
        out = postprocess_mod._remove_attributes_sections(self._page())
        # The first class's detail heading must NOT be eaten.
        assert "#### `CVScheme`" in out
        assert "### Classes" in out
        # No heading-depth jump: h2 is followed by h3, never straight to h5.
        assert "## `cv`\n\n##### Methods" not in out

    def test_attributes_detail_removed(self, postprocess_mod):
        out = postprocess_mod._remove_attributes_sections(self._page())
        # Both the module-level and class-level Attributes detail go away...
        assert "### Attributes" not in out
        assert "##### Attributes" not in out
        assert "#### `CVSchemeType`" not in out
        assert "###### `k`" not in out
        # ...but Methods and its members stay.
        assert "##### Methods" in out
        assert "###### `split`" in out


class TestFixLeadingEmptyTableCells:
    """A table row whose first cell is empty must keep its column alignment.

    griffe2md renders rows without a leading pipe, so a row like
    `` | None`` (empty Type, description "None") loses its first cell — Markdown
    treats the leading pipe as the optional row delimiter and shifts "None" into
    the Type column. Prepending a pipe restores the empty cell.
    """

    def test_empty_first_cell_gets_leading_pipe(self, postprocess_mod):
        page = (
            "**Returns:**\n\n"
            "Type | Description\n"
            "---- | -----------\n"
            " | None (renders inline)\n"
        )
        out = postprocess_mod._fix_leading_empty_table_cells(page)
        assert "\n| | None (renders inline)\n" in out

    def test_populated_rows_and_headers_untouched(self, postprocess_mod):
        page = (
            "Name | Type | Description | Default\n"
            "---- | ---- | ----------- | -------\n"
            "`x` | <code>int</code> | The x. | *required*\n"
        )
        assert postprocess_mod._fix_leading_empty_table_cells(page) == page

    def test_pipes_outside_tables_untouched(self, postprocess_mod):
        page = "```python\n| a |\n```\n\nProse with | a pipe.\n"
        assert postprocess_mod._fix_leading_empty_table_cells(page) == page


class TestRemoveModulesSummary:
    """The ``**Modules:**`` summary table goes away.

    Submodules are not rendered on the package page (``show_submodules`` is off;
    each has its own page), so the table's same-page links dangle — or worse,
    land on a class whose slug matches the module name (`srm` -> `SRM`).
    """

    def test_modules_block_removed_other_blocks_kept(self, postprocess_mod):
        page = (
            "Doc.\n\n"
            "**Methods:**\n\n"
            "Name | Description\n---- | -----------\n"
            "[`align`](#align) | Align.\n\n"
            "**Modules:**\n\n"
            "Name | Description\n---- | -----------\n"
            "[`srm`](#srm) | Shared Response Model.\n"
            "`local` | Local alignment.\n\n"
            "## Classes\n"
        )
        out = postprocess_mod._remove_modules_summary(page)
        assert "**Modules:**" not in out
        assert "`srm`" not in out and "`local`" not in out
        assert "[`align`](#align) | Align.\n\n## Classes\n" in out

    def test_trailing_modules_block_removed(self, postprocess_mod):
        page = (
            "Doc.\n\n**Modules:**\n\nName | Description\n---- | -----------\n`m` | M.\n"
        )
        assert postprocess_mod._remove_modules_summary(page) == "Doc.\n"


class TestPageTitle:
    """Each page's frontmatter title is derived from its import path.

    griffe2md's root heading is off (it duplicated the page title), so the title
    must come from the build script. Bare module names collide across facades
    (`io` appears four times), so modules use their dotted path; classes use
    their name.
    """

    def test_class_uses_bare_name(self, postprocess_mod):
        assert (
            postprocess_mod.page_title("nltools.data.braindata.BrainData")
            == "BrainData"
        )

    def test_module_uses_dotted_path_without_package(self, postprocess_mod):
        assert (
            postprocess_mod.page_title("nltools.data.braindata.io")
            == "data.braindata.io"
        )

    def test_top_level_module(self, postprocess_mod):
        assert postprocess_mod.page_title("nltools.plotting") == "plotting"

    def test_frontmatter_prepended(self, postprocess_mod):
        out = postprocess_mod.with_frontmatter("Body.\n", "data.braindata.io")
        assert out == "---\ntitle: data.braindata.io\n---\n\nBody.\n"


class TestRemoveEmptyCategoryHeadings:
    """`### Classes` / `#### Methods` with nothing under them are dropped.

    griffe2md emits a category heading per member kind even when every member
    of that kind was filtered out, leaving a bare heading followed directly by
    its sibling (or the end of the page).
    """

    def test_empty_heading_before_sibling_removed(self, postprocess_mod):
        page = "## `m`\n\n### Classes\n\n### Methods\n\n#### `f`\n\nDoc.\n"
        out = postprocess_mod._remove_empty_category_headings(page)
        assert "### Classes" not in out
        assert "### Methods\n\n#### `f`" in out

    def test_empty_trailing_heading_removed(self, postprocess_mod):
        page = "## `m`\n\n### Methods\n\n#### `f`\n\nDoc.\n\n### Modules\n"
        out = postprocess_mod._remove_empty_category_headings(page)
        assert "### Modules" not in out
        assert out.endswith("Doc.\n")

    def test_populated_heading_kept(self, postprocess_mod):
        page = "## `m`\n\n### Classes\n\n#### `C`\n\nDoc.\n\n### Methods\n\n#### `f`\n"
        assert postprocess_mod._remove_empty_category_headings(page) == page
