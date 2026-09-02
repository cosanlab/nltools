#!/usr/bin/env python3
"""Post-process griffe2md output into mystmd-clean API pages.

Extracted from `build_api_docs.py` so the postprocess logic — a pipeline of
pure ``str -> str`` (and line-based ``list[str]``) passes — lives in one
testable place. `build_api_docs.generate` calls `postprocess` on each freshly
generated page; this module can also be run standalone on already-generated
files (e.g. to re-apply a postprocess tweak without a full griffe2md regen):

    python scripts/postprocess_api_docs.py docs/api/**/*.md

The per-page label prefix (see `_scope_anchors`) is derived from each file's
path relative to ``docs/api`` via `page_prefix`, so standalone runs and the
in-process `generate` path produce identical output.

Design note: nltools docstrings are Google-style Markdown by policy, so most of
this is a safety net over griffe2md quirks and third-party (re-export) leakage.
The anchor handling deliberately makes every cross-reference *explicit*
(`_scope_anchors` labels headings and prefixes links) to eliminate mystmd
"implicit heading reference" warnings, rather than leaving links implicit.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Repo ``docs/api`` root, used to derive per-page label prefixes. Kept in sync
# with build_api_docs.DOCS_API (both resolve from this file's location).
DOCS_API = Path(__file__).resolve().parent.parent / "docs" / "api"


def _strip_rst_roles(text: str) -> str:
    """Convert any residual RST cross-reference roles to Markdown code spans.

    Docstrings are standardized to Google-style Markdown (no RST), but this is a
    safety net so stray ``:func:`x```-style roles never leak into rendered docs.
    Drops the role name, any ``~`` prefix, and the module path, keeping the short
    symbol (``Class.method`` when the penultimate segment is capitalized).
    """

    def repl(m: re.Match) -> str:
        target = m.group(2).lstrip("~")
        parts = target.split(".")
        if len(parts) >= 2 and parts[-2][:1].isupper():
            short = ".".join(parts[-2:])
        else:
            short = parts[-1]
        return f"`{short}`"

    return re.sub(r":(func|meth|class|attr|obj|mod|data|exc|ref):`([^`]+)`", repl, text)


# RST block directive (``.. name:: args`` + optional deeper-indented body). A
# safety net: nltools no longer re-exports third-party callables (the hrf module
# wraps nilearn's HRFs behind its own Markdown docstrings), but a future re-export
# would leak RST like ``.. nilearn_deprecated:: 0.11.0`` into the page. The body
# is every following line that is blank or indented deeper than the marker, per
# RST; consume the marker and its whole block.
_RST_DIRECTIVE_RE = re.compile(
    r"^(?P<indent>[ \t]*)\.\. [\w-]+::[^\n]*\n"  # ``.. name:: args`` marker line
    r"(?:[ \t]*\n|(?P=indent)[ \t]+[^\n]*\n)*",  # blank or deeper-indented body
    re.MULTILINE,
)


def _strip_rst_directives(text: str) -> str:
    """Drop residual RST block directives leaked from third-party docstrings.

    nltools' own docstrings are Markdown-only, so this only matters if a
    third-party callable is ever re-exported directly (its RST directives, e.g.
    ``.. nilearn_deprecated:: 0.11.0``, would render as literal text).
    Removes the ``.. name::`` marker and any deeper-indented directive body.
    """

    return _RST_DIRECTIVE_RE.sub("", text)


_HEADING_RE = re.compile(r"^(#{2,6}) ")

# A griffe2md summary-table label line, e.g. ``**Methods:**`` / ``**Attributes:**``.
_BLOCK_MARKER_RE = re.compile(r"^\*\*(\w[\w ]*?):\*\*\s*$")

# Canonical order for member summary tables. griffe emits Functions before
# Attributes (and Modules before Classes on module pages); this reorders each
# run to a more natural reading order. Names not listed keep their relative
# order and sort after the listed ones (stable sort).
_SUMMARY_ORDER = [
    "Parameters",
    "Attributes",
    "Classes",
    "Functions",
    "Methods",
    "Modules",
]


def _summary_priority(name: str) -> int:
    """Sort key for a summary block; unlisted names sort last, keeping their order."""
    return _SUMMARY_ORDER.index(name) if name in _SUMMARY_ORDER else len(_SUMMARY_ORDER)


def _reorder_summary_blocks(text: str) -> str:
    """Reorder each run of summary tables to Parameters -> Attributes -> ... -> Modules.

    griffe2md emits the member summary blocks (``**Methods:**``, ``**Attributes:**``,
    ``**Classes:**``, ``**Modules:**``) in a less natural order — Functions before
    Attributes, Modules before Classes. This rewrites each *contiguous run* of
    ``**X:**`` blocks (a run is bounded by any heading line or EOF) into
    `_SUMMARY_ORDER`.

    Adapted from bossanova's `_reorder_summary_blocks`: bossanova demotes heading
    levels so all summary tables land in a single pre-``## `` preamble, but nltools
    keeps griffe's heading levels, so summary tables sit *after* each object heading
    (module ``## ``, class ``#### ``). Operating per-run instead of per-preamble
    covers both, and leaves per-member ``Parameters -> Returns -> Examples`` runs
    untouched (Parameters is already first there, so the stable sort is a no-op).
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if not _BLOCK_MARKER_RE.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        # Collect a maximal run of consecutive summary blocks. Each block spans
        # its marker line through the content up to (not including) the next
        # marker or heading line.
        run_start = i
        blocks: list[tuple[str, list[str]]] = []
        while i < n and (m := _BLOCK_MARKER_RE.match(lines[i])):
            start = i
            i += 1
            while (
                i < n
                and not _BLOCK_MARKER_RE.match(lines[i])
                and not _HEADING_RE.match(lines[i])
            ):
                i += 1
            blocks.append((m.group(1), lines[start:i]))
        if len(blocks) < 2:
            out.extend(lines[run_start:i])
            continue
        for _, block_lines in sorted(blocks, key=lambda b: _summary_priority(b[0])):
            out.extend(block_lines)
    return "\n".join(out)


def _remove_modules_summary(text: str) -> str:
    """Drop the ``**Modules:**`` summary table from package pages.

    Submodules aren't rendered on the package page (``show_submodules`` is off;
    each has its own page), so the table's same-page links either dangle or hit
    a class whose slug matches the module name (``srm`` -> ``SRM``). The block
    spans its marker line to the next summary marker, heading, or EOF.
    """
    block = re.compile(
        r"^\*\*Modules:\*\*\s*\n(?:(?!\*\*\w[\w ]*?:\*\*\s*$|#{2,6} ).*\n?)*",
        flags=re.MULTILINE,
    )
    text = block.sub("", text)
    return re.sub(r"\n{2,}\Z", "\n", text)


def _remove_attributes_sections(text: str) -> str:
    """Drop every ``Attributes`` detail section, keeping the summary table.

    An Attributes section at heading level ``L`` spans from its heading to the
    next heading of level ``<= L`` (its next sibling or a shallower heading).
    Terminating on level — not merely on "the next Methods heading" — is what
    keeps a module-level ``### Attributes`` section from greedily swallowing the
    following ``### Classes`` / ``#### FirstClass`` headings, which collapsed the
    page into a module-h2 -> class-Methods-h5 jump that mystmd warned on.

    Assumes concatenated headings have already been split onto their own lines
    (see the ``postprocess`` step that runs before this), so each heading —
    including ``#### Attributes`` — is on a line of its own.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _HEADING_RE.match(line)
        if m and line[len(m.group(1)) + 1 :].strip() == "Attributes":
            level = len(m.group(1))
            # Skip this heading and everything under it until a heading of
            # level <= this one (sibling/parent), which terminates the section.
            i += 1
            while i < n:
                mm = _HEADING_RE.match(lines[i])
                if mm and len(mm.group(1)) <= level:
                    break
                i += 1
            # Collapse any blank lines left dangling before the terminator so we
            # don't accumulate a widening gap where the section used to be.
            while out and out[-1] == "":
                out.pop()
            out.append("")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


_CATEGORY_HEADINGS = {"Attributes", "Classes", "Methods", "Functions", "Modules"}
_TABLE_SEPARATOR_RE = re.compile(r"^\s*:?-{3,}:?(\s*\|\s*:?-{3,}:?)+\s*\|?\s*$")


def _remove_empty_category_headings(text: str) -> str:
    """Drop member-category headings (``### Classes`` ...) with nothing under them.

    griffe2md emits one category heading per member kind even when every member
    of that kind was filtered out (private, deprecated, hidden), leaving a bare
    heading followed directly by a heading of the same-or-shallower level or by
    the end of the page.
    """
    lines = text.split("\n")
    out: list[str] = []
    just_removed = False
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m and line[len(m.group(1)) + 1 :].strip() in _CATEGORY_HEADINGS:
            level = len(m.group(1))
            following = [ln for ln in lines[i + 1 :] if ln.strip()]
            nxt = _HEADING_RE.match(following[0]) if following else None
            if not following or (nxt and len(nxt.group(1)) <= level):
                # Drop the heading; the blank lines around it collapse to one.
                while out and out[-1] == "":
                    out.pop()
                out.append("")
                just_removed = True
                continue
        if just_removed and line == "":
            continue
        just_removed = False
        out.append(line)
    if len(out) > 1 and out[-1] == "" and out[-2] == "":
        out.pop()
    return "\n".join(out)


def _fix_leading_empty_table_cells(text: str) -> str:
    """Keep column alignment for table rows whose first cell is empty.

    griffe2md writes rows without a leading pipe, so a Returns row with no type
    (``" | None"``) reads to Markdown as a single cell — the leading pipe is the
    optional row delimiter — and the description shifts into the Type column.
    Prepending a pipe turns it back into an explicit empty first cell.
    """
    lines = text.split("\n")
    in_table = in_fence = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            in_table = False
        elif _TABLE_SEPARATOR_RE.match(line):
            in_table = True
        elif in_table and line.lstrip().startswith("|"):
            lines[i] = "|" + line
    return "\n".join(lines)


def _remove_deprecated_members(text: str) -> str:
    """Hide deprecated methods from the generated API reference.

    Self-targeting: only removes members whose docstring summary begins with
    ``Deprecated``. Deprecations are documented in the migration guide, not the
    API reference. Removes both the Methods summary-table row and the detail
    section. Deprecated *parameter aliases* (table rows like ``threshold``) are
    untouched — they are not method rows and their bodies don't open with
    ``Deprecated:`` after a signature fence.
    """
    # Drop Methods summary-table rows: [`name`](#anchor) | Deprecated...
    # (anchors are slugified by now, so the char class must include `-`.)
    text = re.sub(
        r"^\[`\w+`\]\(#[\w.-]+\) \| Deprecated.*\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Drop detail sections whose body opens with "Deprecated:" right after the
    # signature code fence. Section spans "#### `name`" to the next heading/EOF.
    def _drop(m: re.Match) -> str:
        body = m.group(0)
        return "" if re.search(r"```\n+Deprecated:", body) else body

    return re.sub(
        r"^#### `\w+`\n.*?(?=^#### |^### |^## |\Z)",
        _drop,
        text,
        flags=re.MULTILINE | re.DOTALL,
    )


def _myst_slug(heading_text: str) -> str:
    """Reproduce mystmd's implicit heading-id slug for a heading's text.

    Lowercase, drop inline formatting, and collapse any run of non-alphanumeric
    characters to a single hyphen (so ``\`one_sample_permutation_test\``` -> id
    ``one-sample-permutation-test``, matching what mystmd records in its xref).
    """
    plain = re.sub(r"[`*]", "", heading_text)
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def _scope_anchors(text: str, prefix: str) -> str:
    """Give each member heading an explicit, page-scoped MyST target label.

    mystmd warns ("implicit heading reference") whenever a cross-reference
    resolves to a heading's auto-generated id instead of an explicit label.
    griffe2md links every summary-table row to a member heading by its slug, so
    without explicit labels each linked member emits one such warning (1000+ per
    full build). The fix is to label the headings and reference the labels.

    The catch: explicit MyST labels are project-*global*, whereas the implicit
    heading ids they replace are page-*local*. Common method names repeat across
    facades (``align`` is a `stats` function, a `BrainData` method, ...), so a
    bare ``(align)=`` on every page collides — trading implicit-ref warnings for
    just as many "Duplicate identifier" ones. Prefixing every label and its
    same-page links with a per-page ``prefix`` (derived from the output path)
    restores page-scoped uniqueness while keeping the reference explicit.

    Two coordinated edits, using the same first-occurrence slug set:
      1. Inject ``(prefix-slug)=`` before the first code-span heading for each
         slug (``#### `name` `` — the members/submodules summary tables link to;
         section headings like Methods/Attributes are never linked, so skipped).
      2. Rewrite each surviving same-page link ``](#slug)`` -> ``](#prefix-slug)``
         for slugs that got a label, so links still resolve to their heading.

    Runs last in `postprocess`, after `_delink_dangling_anchors` has dropped any
    link whose heading was removed — so every remaining ``](#slug)`` targets a
    real on-page heading.
    """
    labeled: set[str] = set()
    out: list[str] = []
    heading_re = re.compile(r"^#{1,6}\s+`([^`]+)`\s*$")
    for line in text.split("\n"):
        m = heading_re.match(line)
        if m:
            slug = _myst_slug(m.group(1))
            if slug and slug not in labeled:
                labeled.add(slug)
                out.append(f"({prefix}-{slug})=")
        out.append(line)
    text = "\n".join(out)

    def relink(m: re.Match) -> str:
        slug = m.group(1)
        return f"](#{prefix}-{slug})" if slug in labeled else m.group(0)

    return re.sub(r"\]\(#([\w-]+)\)", relink, text)


def _delink_dangling_anchors(text: str) -> str:
    """De-link summary-table links whose target heading isn't on the page.

    Every generated link is same-page (``[\`name\`](#slug)``). After we remove
    the Attributes detail section and deprecated members, some summary links point
    at headings that no longer exist. Rather than leave a dangling reference, drop
    the link wrapper and keep the name as a plain code span.
    """
    slugs = {
        _myst_slug(m.group(1))
        for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    }

    def repl(m: re.Match) -> str:
        return m.group(0) if m.group(2) in slugs else f"`{m.group(1)}`"

    return re.sub(r"\[`([^`]+)`\]\(#([\w.-]+)\)", repl, text)


def postprocess(text: str, prefix: str) -> str:
    """Fix griffe2md output quirks.

    - Insert newline between concatenated headings (e.g. ``### Attributes#### foo``)
    - Shorten dotpath anchors in summary table links to match short heading IDs
    - Rename 'Functions' summary/category heading to 'Methods' for class pages
    - Strip any residual RST roles and block directives (safety net over
      docstring standardization; catches third-party re-export leakage)
    - Hide deprecated members (documented in the migration guide instead)
    - Label member headings with page-scoped explicit MyST targets

    ``prefix`` is a per-page slug (from the output path) that namespaces the
    explicit heading labels so they stay unique project-wide.
    """
    # Fix concatenated headings: "### Foo#### Bar" -> "### Foo\n\n#### Bar"
    text = re.sub(r"(#{2,6} .+?)(#{2,6} )", r"\1\n\n\2", text)

    # Shorten summary-table anchor links to the member's MyST heading id.
    # griffe2md emits full-dotpath anchors, e.g. [`foo_bar`](#pkg.mod.foo_bar),
    # but mystmd slugifies heading text (lowercase, `_` -> `-`), so the rendered
    # heading `#### \`foo_bar\`` gets id `foo-bar`. Match that slug or every member
    # whose name has an underscore or capital letter yields a dangling link.
    def _shorten_anchor(m: re.Match) -> str:
        name = m.group(2)
        slug = name.lower().replace("_", "-")
        return f"[`{m.group(1)}`](#{slug})"

    text = re.sub(r"\[`(\w+)`\]\(#[\w.]+\.(\w+)\)", _shorten_anchor, text)

    # Rename "Functions" to "Methods" in category headings and summary labels
    text = re.sub(r"^(#{2,6}) Functions$", r"\1 Methods", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\*Functions:\*\*$", "**Methods:**", text, flags=re.MULTILINE)

    # Submodules have their own pages, so the Modules summary table only offers
    # links that dangle (or hit a same-slug class); drop it, then reorder the
    # remaining summary tables to a natural reading order (griffe emits
    # Functions before Attributes). Runs after the Functions->Methods rename so
    # blocks carry canonical labels.
    text = _remove_modules_summary(text)
    text = _reorder_summary_blocks(text)

    # Remove "Bases: object" (noise for classes that only inherit from object)
    text = re.sub(r"\nBases: <code>\[object\]\(#object\)</code>\n", "\n", text)

    # Remove every Attributes detail section (heading + individual entries); the
    # summary table is enough. Level-aware: each section ends at the next heading
    # of same-or-shallower level, so a module-level `### Attributes` can't eat the
    # following `### Classes` / `#### FirstClass` headings (which produced a
    # module-h2 -> class-Methods-h5 depth jump that mystmd warned on).
    text = _remove_attributes_sections(text)

    # Hide deprecated members and strip any residual RST roles (safety net).
    text = _remove_deprecated_members(text)
    text = _strip_rst_roles(text)
    text = _strip_rst_directives(text)

    # Category headings left empty by the removals above (or by griffe2md's own
    # member filtering) are dropped; rows whose first cell is empty get an
    # explicit leading pipe so their columns stay aligned.
    text = _remove_empty_category_headings(text)
    text = _fix_leading_empty_table_cells(text)

    # Final pass: drop summary links whose target heading was removed above.
    text = _delink_dangling_anchors(text)

    # Label the surviving member/submodule headings (and their same-page links)
    # with page-scoped explicit targets, so references resolve explicitly.
    text = _scope_anchors(text, prefix)

    return text


def page_prefix(output: Path, docs_api: Path = DOCS_API) -> str:
    """Per-page label prefix: the output path relative to ``docs_api``, slugified.

    Used by `_scope_anchors` to namespace explicit heading labels so common
    member names (``align``, ``fit``, ...) don't collide project-wide. Unique
    per page because output paths are unique.
    """
    rel = output.resolve().relative_to(docs_api.resolve()).with_suffix("")
    return _myst_slug(rel.as_posix())


def page_title(module: str) -> str:
    """Frontmatter title for the page documenting ``module`` (an import path).

    griffe2md's root heading is disabled (it repeated the page title), so the
    title comes from here. A class page is titled by the class name; a module
    page by its dotted path minus the ``nltools.`` prefix, because bare module
    names repeat across facades (``io`` alone would title four pages).
    """
    parts = module.split(".")
    if parts[-1][:1].isupper():
        return parts[-1]
    return ".".join(parts[1:]) if parts[0] == "nltools" else module


def with_frontmatter(text: str, title: str) -> str:
    """Prepend a MyST frontmatter block carrying ``title``."""
    return f"---\ntitle: {title}\n---\n\n{text}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="+", help="Generated API markdown files to post-process"
    )
    parser.add_argument(
        "--docs-api",
        type=Path,
        default=DOCS_API,
        help="Root the per-page label prefix is derived against (default: repo docs/api)",
    )
    args = parser.parse_args()

    for path in args.files:
        output = Path(path)
        text = output.read_text()
        output.write_text(postprocess(text, page_prefix(output, args.docs_api)))
        print(f"  post-processed {path}")


if __name__ == "__main__":
    main()
