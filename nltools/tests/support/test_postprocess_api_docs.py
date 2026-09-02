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


class TestStripRootHeading:
    """griffe2md's root heading goes; the constructor signature fence stays.

    ``show_root_heading`` is on only so the class template emits the class's
    signature fence (it is skipped otherwise). The heading itself repeats the
    frontmatter title, so the one level-1 heading at the top of the page is
    dropped, and nothing else that starts with ``#`` (fenced example comments)
    is touched.
    """

    def test_class_root_heading_removed_signature_kept(self, postprocess_mod):
        page = (
            "# `BrainData`\n\n"
            "```python\nBrainData(data=None, *, mask=None)\n```\n\n"
            "Doc.\n\n"
            "```python\n# Hyperalign using procrustes transform\nbd.align(x)\n```\n"
        )
        out = postprocess_mod._strip_root_heading(page)
        assert not out.startswith("#")
        assert out.startswith("```python\nBrainData(data=None, *, mask=None)\n```")
        assert "# Hyperalign using procrustes transform" in out

    def test_module_root_heading_removed(self, postprocess_mod):
        page = "# `corrections`\n\nModule doc.\n\n## Functions\n"
        out = postprocess_mod._strip_root_heading(page)
        assert out == "Module doc.\n\n## Functions\n"

    def test_page_without_root_heading_untouched(self, postprocess_mod):
        page = "Module doc.\n\n## Functions\n"
        assert postprocess_mod._strip_root_heading(page) == page


class TestClassSections:
    """`Functions` -> `Methods` and ``-> None`` stripping are scoped to classes.

    griffe2md labels every function group ``Functions``; only inside a class
    (a class page, or a ``### `ClassName``` section on a module page) are those
    functions methods. Module-level functions keep the ``Functions`` label.
    The class's own signature fence (merged ``__init__``) ends in ``-> None``,
    which is noise for a constructor; a method that returns None keeps it.
    """

    def test_class_page_renames_functions(self, postprocess_mod):
        page = (
            "# `BrainData`\n\n"
            "```python\nBrainData(data=None, *, mask=None) -> None\n```\n\n"
            "**Functions:**\n\n"
            "Name | Description\n---- | -----------\n[`align`](#align) | Align.\n\n"
            "## Functions\n\n"
            "### `align`\n\n```python\nalign(target) -> None\n```\n"
        )
        out = postprocess_mod._apply_class_sections(page)
        assert "**Methods:**" in out and "**Functions:**" not in out
        assert "## Methods" in out and "## Functions" not in out
        # Constructor loses the `-> None`; the method keeps its real return.
        assert "BrainData(data=None, *, mask=None)\n```" in out
        assert "align(target) -> None" in out

    def test_module_page_keeps_functions(self, postprocess_mod):
        page = (
            "# `corrections`\n\n"
            "**Functions:**\n\n"
            "Name | Description\n---- | -----------\n[`fdr`](#fdr) | FDR.\n\n"
            "## Functions\n\n"
            "### `fdr`\n\n```python\nfdr(p, q=0.05)\n```\n"
        )
        assert postprocess_mod._apply_class_sections(page) == page

    def test_module_page_with_classes_renames_only_inside_classes(
        self, postprocess_mod
    ):
        page = (
            "# `models`\n\n"
            "## Classes\n\n"
            "### `Glm`\n\n"
            "```python\nGlm(*, t_r: float | None = None) -> None\n```\n\n"
            "**Functions:**\n\n"
            "Name | Description\n---- | -----------\n[`fit`](#fit) | Fit.\n\n"
            "#### Functions\n\n"
            "##### `fit`\n\n```python\nfit(X, y) -> None\n```\n\n"
            "## Functions\n\n"
            "### `helper`\n\n```python\nhelper() -> None\n```\n"
        )
        out = postprocess_mod._apply_class_sections(page)
        assert "#### Methods\n\n##### `fit`" in out
        assert "**Methods:**" in out
        assert "## Functions\n\n### `helper`" in out
        assert "Glm(*, t_r: float | None = None)\n```" in out
        assert "fit(X, y) -> None" in out
        assert "helper() -> None" in out


class TestResolveTypeLinks:
    """Type-annotation links become plain text or real cross-page references.

    griffe2md wraps every annotation in a same-page link (``[str](#str)``,
    ``[np.ndarray](#numpy.ndarray)``), none of which resolve. nltools types
    link to the page that documents them (via the build-time xref index);
    anything else is de-linked to plain text.
    """

    XREF = {
        "nltools.data.braindata.BrainData": "data-brain-data",
        "nltools.data.braindata.BrainData.align": "data-brain-data-align",
        "nltools.data.fitresults": "data-fitresults",
        "nltools.data.fitresults.PredictCollection": "data-fitresults-predictcollection",
        "nltools.algorithms.SRM": "algorithms-srm",
        "nltools.algorithms.alignment.SRM": "algorithms-alignment-srm",
        "nltools.models.BaseModel": "models-basemodel",
    }

    def test_builtin_and_third_party_types_delinked(self, postprocess_mod):
        page = "`x` | <code>[str](#str) \\| [np.ndarray](#numpy.ndarray)</code> | X.\n"
        out = postprocess_mod._resolve_type_links(page, self.XREF)
        assert out == "`x` | <code>str \\| np.ndarray</code> | X.\n"

    def test_nltools_type_links_to_its_page(self, postprocess_mod):
        page = "<code>[BrainData](#nltools.data.braindata.BrainData)</code> | Out."
        out = postprocess_mod._resolve_type_links(page, self.XREF)
        assert out == "<code>[BrainData](#data-brain-data)</code> | Out."

    def test_bare_name_resolves_by_suffix(self, postprocess_mod):
        # griffe couldn't resolve the annotation (string/TYPE_CHECKING import),
        # so the anchor is the bare name.
        page = "<code>[PredictCollection](#PredictCollection)</code>"
        out = postprocess_mod._resolve_type_links(page, self.XREF)
        assert (
            out
            == "<code>[PredictCollection](#data-fitresults-predictcollection)</code>"
        )

    def test_canonical_path_prefers_closest_page(self, postprocess_mod):
        # SRM is rendered on two pages; the canonical (defining) path shares the
        # longest prefix with the alignment page.
        page = "[SRM](#nltools.algorithms.alignment.srm.SRM)"
        out = postprocess_mod._resolve_type_links(page, self.XREF)
        assert out == "[SRM](#algorithms-alignment-srm)"

    def test_bare_name_on_two_pages_prefers_deepest(self, postprocess_mod):
        # A facade re-export and the defining subpackage both document SRM; a
        # bare name has no path to disambiguate, so the more specific page wins.
        page = "[SRM](#SRM)"
        out = postprocess_mod._resolve_type_links(page, self.XREF)
        assert out == "[SRM](#algorithms-alignment-srm)"

    def test_unknown_nltools_symbol_delinked(self, postprocess_mod):
        page = "[Secret](#nltools.internal._Secret)"
        assert postprocess_mod._resolve_type_links(page, self.XREF) == "Secret"

    def test_without_index_everything_is_plain(self, postprocess_mod):
        page = "[BrainData](#nltools.data.braindata.BrainData) and [str](#str)"
        assert postprocess_mod._resolve_type_links(page, None) == "BrainData and str"

    def test_same_page_member_links_untouched(self, postprocess_mod):
        # Backticked summary-table links are the labeller's business.
        page = "[`align`](#data-brain-data-align) | Align."
        assert postprocess_mod._resolve_type_links(page, self.XREF) == page

    def test_raises_table_type_column(self, postprocess_mod):
        page = (
            "**Raises:**\n\n"
            "Type | Description\n---- | -----------\n"
            "<code>[ValueError](#ValueError)</code> | If ``x`` is empty.\n"
        )
        out = postprocess_mod._resolve_type_links(page, self.XREF)
        assert "<code>ValueError</code> | If ``x`` is empty." in out


class TestCleanBases:
    """``Bases:`` lines lose private and ``object`` bases; the rest are readable.

    Third-party bases can't be linked (no page), so they render as their full
    dotted path in a code span; nltools bases link to their page; private bases
    (``_BaseKFold``) are implementation detail and are dropped, along with the
    whole line when nothing public remains.
    """

    XREF = {"nltools.models.BaseModel": "models-basemodel"}

    def test_private_base_dropped_with_line(self, postprocess_mod):
        page = (
            "### `KFold`\n\n"
            "Bases: <code>[_BaseKFold](#sklearn.model_selection._split._BaseKFold)</code>\n\n"
            "Doc.\n"
        )
        out = postprocess_mod._clean_bases(page, self.XREF)
        assert out == "### `KFold`\n\nDoc.\n"

    def test_object_base_dropped(self, postprocess_mod):
        page = "### `C`\n\nBases: <code>[object](#object)</code>\n\nDoc.\n"
        assert postprocess_mod._clean_bases(page, self.XREF) == "### `C`\n\nDoc.\n"

    def test_third_party_bases_as_plain_paths(self, postprocess_mod):
        page = (
            "Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, "
            "<code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>\n"
        )
        out = postprocess_mod._clean_bases(page, self.XREF)
        assert out == (
            "Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`\n"
        )

    def test_nltools_base_links_to_page(self, postprocess_mod):
        page = "Bases: <code>[BaseModel](#nltools.models.base.BaseModel)</code>\n"
        out = postprocess_mod._clean_bases(page, self.XREF)
        assert out == "Bases: [`BaseModel`](#models-basemodel)\n"

    def test_mixed_keeps_public_only(self, postprocess_mod):
        page = (
            "Bases: <code>[_Private](#pkg._Private)</code>, "
            "<code>[ABC](#abc.ABC)</code>\n"
        )
        assert postprocess_mod._clean_bases(page, self.XREF) == "Bases: `abc.ABC`\n"


class TestRemoveDeprecatedMembers:
    """Deprecated members are hidden at every heading depth griffe2md emits.

    Members sit at ``### `` on module and class pages, at ``##### `` for
    methods of a class on a module page. The old regex only matched ``#### ``,
    so it was dormant. The summary row goes too.
    """

    @pytest.mark.parametrize("hashes", ["###", "####", "#####"])
    def test_deprecated_member_removed_at_each_depth(self, postprocess_mod, hashes):
        page = (
            "**Methods:**\n\n"
            "Name | Description\n---- | -----------\n"
            "[`old`](#old) | Deprecated: use `new`.\n"
            "[`new`](#new) | The new one.\n\n"
            f"{hashes} `new`\n\n```python\nnew()\n```\n\nThe new one.\n\n"
            f"{hashes} `old`\n\n```python\nold()\n```\n\nDeprecated: use `new`.\n\n"
            f"{hashes} `other`\n\n```python\nother()\n```\n\nKept.\n"
        )
        out = postprocess_mod._remove_deprecated_members(page)
        assert f"{hashes} `old`" not in out
        assert "Deprecated: use `new`." not in out
        assert "[`old`](#old)" not in out
        assert f"{hashes} `new`" in out and f"{hashes} `other`" in out
        assert "[`new`](#new) | The new one." in out

    def test_non_deprecated_untouched(self, postprocess_mod):
        page = "### `f`\n\n```python\nf()\n```\n\nNormal.\n"
        assert postprocess_mod._remove_deprecated_members(page) == page


class TestXrefEntries:
    """The xref index maps dotted symbol paths to page-scoped MyST labels.

    Built from each postprocessed page: the page itself (its frontmatter
    label, in the ``page-`` namespace), each labelled member, and class-nested
    members as ``Class.member``. Module pages derive member paths from the
    page's ``module``; task pages composed from several modules pass ``roots``
    (top-level heading name -> the dotted paths that heading documents).
    """

    def test_module_page_entries(self, postprocess_mod):
        page = (
            "Doc.\n\n## Classes\n\n"
            "(models-glm)=\n### `Glm`\n\nDoc.\n\n#### Methods\n\n"
            "(models-fit)=\n##### `fit`\n\nDoc.\n\n"
            "## Functions\n\n(models-helper)=\n### `helper`\n\nDoc.\n"
        )
        entries = postprocess_mod.xref_entries("models", page, module="nltools.models")
        assert entries == {
            "nltools.models": "page-models",
            "nltools.models.Glm": "models-glm",
            "nltools.models.Glm.fit": "models-fit",
            "nltools.models.helper": "models-helper",
        }

    def test_class_page_entries(self, postprocess_mod):
        page = "Doc.\n\n## Methods\n\n(data-brain-data-align)=\n### `align`\n\nDoc.\n"
        entries = postprocess_mod.xref_entries(
            "data-brain-data", page, module="nltools.data.braindata.BrainData"
        )
        assert entries == {
            "nltools.data.braindata.BrainData": "page-data-brain-data",
            "nltools.data.braindata.BrainData.align": "data-brain-data-align",
        }

    def test_task_page_entries_use_roots(self, postprocess_mod):
        # A task page mixes objects from several modules; every path listed
        # for a heading (public re-export and canonical definition) maps to
        # that heading's label, and class members nest under each of them.
        page = (
            "Intro.\n\n## Classes\n\n"
            "(tasks-alignment-srm)=\n### `SRM`\n\nDoc.\n\n#### Methods\n\n"
            "(tasks-alignment-fit)=\n##### `fit`\n\nDoc.\n\n"
            "## Functions\n\n(tasks-alignment-align)=\n### `align`\n\nDoc.\n"
        )
        roots = {
            "SRM": ["nltools.algorithms.SRM", "nltools.algorithms.alignment.srm.SRM"],
            "align": ["nltools.algorithms.align"],
        }
        entries = postprocess_mod.xref_entries("tasks-alignment", page, roots=roots)
        assert entries == {
            "nltools.algorithms.SRM": "tasks-alignment-srm",
            "nltools.algorithms.alignment.srm.SRM": "tasks-alignment-srm",
            "nltools.algorithms.SRM.fit": "tasks-alignment-fit",
            "nltools.algorithms.alignment.srm.SRM.fit": "tasks-alignment-fit",
            "nltools.algorithms.align": "tasks-alignment-align",
        }

    def test_heading_without_a_root_or_module_is_skipped(self, postprocess_mod):
        page = "(p-x)=\n### `x`\n\nDoc.\n"
        assert postprocess_mod.xref_entries("p", page) == {}


class TestPageLabel:
    """Page labels live in their own ``page-`` namespace.

    Regression: the label of ``api/data/design_matrix_append.md`` (its prefix,
    ``data-design-matrix-append``) collided with the member label for
    ``DesignMatrix.append`` on ``api/data/design_matrix.md`` (prefix
    ``data-design-matrix`` + slug ``append``) — mystmd reported a duplicate
    identifier. Prefixing page labels keeps the two namespaces apart.
    """

    def test_page_label_is_namespaced(self, postprocess_mod):
        assert postprocess_mod.page_label("data-brain-data") == "page-data-brain-data"

    def test_page_label_cannot_collide_with_member_label(self, postprocess_mod):
        member = postprocess_mod._scope_anchors("### `append`\n", "data-design-matrix")
        member_label = postprocess_mod.collect_labels(member)
        assert member_label == ["data-design-matrix-append"]
        assert (
            postprocess_mod.page_label("data-design-matrix-append") not in member_label
        )

    def test_frontmatter_label_written_when_given(self, postprocess_mod):
        out = postprocess_mod.with_frontmatter(
            "Body.\n", "BrainData", "page-data-brain-data"
        )
        assert (
            out == "---\ntitle: BrainData\nlabel: page-data-brain-data\n---\n\nBody.\n"
        )


class TestCollectLabels:
    """Every explicit MyST label a page defines: frontmatter + ``(x)=`` targets."""

    def test_frontmatter_and_targets(self, postprocess_mod):
        page = (
            "---\ntitle: T\nlabel: page-t\n---\n\n"
            "(t-a)=\n### `a`\n\nDoc with (not-a-label)= in prose.\n\n"
            "```python\n(fenced)=\n```\n\n(t-b)=\n### `b`\n"
        )
        assert postprocess_mod.collect_labels(page) == ["page-t", "t-a", "t-b"]


class TestDocumentedNames:
    """Names a page documents: code-span headings and summary-table rows."""

    def test_headings_and_summary_rows(self, postprocess_mod):
        page = (
            "**Attributes:**\n\n"
            "Name | Type | Description\n---- | ---- | -----------\n"
            "`ATLASES` | <code>dict</code> | Registry.\n\n"
            "**Functions:**\n\n"
            "Name | Description\n---- | -----------\n"
            "[`fdr`](#tasks-inference-fdr) | FDR.\n\n"
            "## Classes\n\n(tasks-x-glm)=\n### `Glm`\n\n"
            "```python\nGlm(*, t_r=None)\n```\n\n"
            "`not_a_name` in prose | is ignored.\n"
        )
        assert postprocess_mod.documented_names(page) == {"ATLASES", "fdr", "Glm"}


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
