#!/usr/bin/env python3
"""Render the canonical-kwarg vocabulary into the docs from one source.

`docs/_data/api-vocabulary.yml` is the single machine-readable source of truth for
the v0.6.0 canonical-kwarg vocabulary. This script renders it into two hand-authored
docs, replacing the content between `<!-- AUTOGEN:api-vocabulary:<block> -->` and
`<!-- /AUTOGEN:api-vocabulary:<block> -->` marker pairs:

  - docs/development/index.md    — block `index-table`   (2-column Markdown table)
  - docs/public/design-tour.html — block `tour-table`    (3-column HTML table body)
                                    block `tour-exceptions` (the "exceptions" callout)

Everything OUTSIDE the markers is left untouched, so the surrounding hand-crafted
prose/markup (the 800-line design tour in particular) is preserved verbatim.

Usage:
    python scripts/build_api_vocabulary.py          # write the rendered blocks in place
    python scripts/build_api_vocabulary.py --check   # exit 1 if any committed block is stale

`--check` is the drift gate: it is wired into `poe lint-api` (run by CI), so an edit
to the vocabulary — or to a generated table by hand — fails the build until the docs
are regenerated. The write mode is wired into `poe docs-generate`.

The HUMAN canon is CLAUDE.md's "API Conventions (v0.6.0)" table; keep the YAML in
sync with it (CLAUDE.md is read directly by Claude and is not generated).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_YML = PROJECT_ROOT / "docs" / "_data" / "api-vocabulary.yml"
INDEX_MD = PROJECT_ROOT / "docs" / "development" / "index.md"
DESIGN_TOUR = PROJECT_ROOT / "docs" / "public" / "design-tour.html"

_NUMBER_WORDS = {
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}


def _load_vocab() -> dict:
    with VOCAB_YML.open() as f:
        return yaml.safe_load(f)


def _md_inline_to_html(text: str) -> str:
    """Convert the limited Markdown used in the YAML to HTML.

    Only `code` spans and *emphasis* appear in the vocabulary notes/exceptions;
    handle exactly those so the design-tour markup matches its hand-authored style.
    """
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def _escape_table_cell(md: str) -> str:
    """Escape pipe characters so a Markdown value set survives inside a table cell."""
    return md.replace("|", r"\|")


def render_index_table(vocab: dict) -> str:
    """Render the 2-column Markdown vocabulary table for docs/development/index.md."""
    lines = ["| Concept | Canonical kwarg |", "|---|---|"]
    for row in vocab["vocabulary"]:
        lines.append(f"| {row['concept']} | {_escape_table_cell(row['index_md'])} |")
    return "\n".join(lines)


def render_tour_table(vocab: dict) -> str:
    """Render the 3-column HTML table body (<tr> rows) for the design tour."""
    rows = []
    for row in vocab["vocabulary"]:
        concept = row["concept"]
        kwarg = f"<code>{row['kwarg']}</code>"
        note = _md_inline_to_html(row["tour_note"])
        rows.append(f"    <tr><td>{concept}</td><td>{kwarg}</td><td>{note}</td></tr>")
    return "\n".join(rows)


def render_tour_exceptions(vocab: dict) -> str:
    """Render the inner HTML of the design-tour "deliberate exceptions" callout."""
    exceptions = vocab["exceptions"]
    count = _NUMBER_WORDS.get(len(exceptions), str(len(exceptions)))
    sentences = " ".join(_md_inline_to_html(e) for e in exceptions)
    return (
        f"  <strong>{count} deliberate exceptions.</strong> {sentences} "
        "Each is documented, not an oversight."
    )


def _marker_pair(block: str) -> tuple[re.Pattern, str]:
    """Return (regex matching the marked region, block name) for an AUTOGEN block.

    The opening marker may carry a trailing human note after the block name; the
    closing marker is exactly `<!-- /AUTOGEN:api-vocabulary:<block> -->`.
    """
    open_re = rf"<!-- AUTOGEN:api-vocabulary:{re.escape(block)}\b.*?-->"
    close_re = rf"<!-- /AUTOGEN:api-vocabulary:{re.escape(block)} -->"
    # Capture the open marker (1), the body (2), and the close marker (3).
    pattern = re.compile(rf"({open_re})(.*?)({close_re})", re.DOTALL)
    return pattern, block


def _replace_block(text: str, block: str, body: str, *, path: Path) -> str:
    """Replace the body between a block's markers, preserving the marker lines."""
    pattern, _ = _marker_pair(block)
    match = pattern.search(text)
    if match is None:
        raise SystemExit(
            f"error: AUTOGEN block '{block}' markers not found in {path}. "
            f"Expected <!-- AUTOGEN:api-vocabulary:{block} ... --> ... "
            f"<!-- /AUTOGEN:api-vocabulary:{block} -->."
        )
    open_marker, _, close_marker = match.group(1), match.group(2), match.group(3)
    # Match the closing marker's indentation to the opening marker's line so the
    # generated region stays visually aligned with the hand-authored markup around
    # it. (For the Markdown table the indent is empty — a table must start at
    # column 0 or it renders as a code block.)
    line_start = text.rfind("\n", 0, match.start()) + 1
    indent = text[line_start : match.start()]
    replacement = f"{open_marker}\n{body}\n{indent}{close_marker}"
    return text[: match.start()] + replacement + text[match.end() :]


def _apply(path: Path, blocks: dict[str, str]) -> str:
    """Return `path`'s text with every named block's body replaced."""
    text = path.read_text()
    for block, body in blocks.items():
        text = _replace_block(text, block, body, path=path)
    return text


def build(check: bool) -> int:
    vocab = _load_vocab()

    targets = {
        INDEX_MD: {"index-table": render_index_table(vocab)},
        DESIGN_TOUR: {
            "tour-table": render_tour_table(vocab),
            "tour-exceptions": render_tour_exceptions(vocab),
        },
    }

    stale: list[Path] = []
    for path, blocks in targets.items():
        rendered = _apply(path, blocks)
        if rendered != path.read_text():
            stale.append(path)
            if not check:
                path.write_text(rendered)

    rel = lambda p: p.relative_to(PROJECT_ROOT)  # noqa: E731
    if check:
        if stale:
            listing = "\n".join(f"  - {rel(p)}" for p in stale)
            print(
                "error: canonical-kwarg vocabulary is stale in:\n"
                f"{listing}\n"
                "Run `uv run poe docs-generate` (or `python scripts/build_api_vocabulary.py`) "
                "after editing docs/_data/api-vocabulary.yml, and commit the result.",
                file=sys.stderr,
            )
            return 1
        print("api-vocabulary: docs are in sync with docs/_data/api-vocabulary.yml.")
        return 0

    if stale:
        for p in stale:
            print(f"  updated {rel(p)}")
    else:
        print("api-vocabulary: docs already in sync; nothing to write.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any generated block is stale (drift gate; no writes).",
    )
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
