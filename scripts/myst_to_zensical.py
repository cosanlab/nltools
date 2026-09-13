"""Convert MyST markdown pages to the Zensical (Python-Markdown + pymdownx) dialect.

One-shot migration tool: run it over a page once, check the result by hand, and
commit the converted markdown. It is not part of the docs build.

Rules:

- Frontmatter keeps `title` and `description` (Zensical reads both) and drops
  every other key, which only MyST understands. A frontmatter block left with no
  keys is removed.
- `:::{note}` / `:::{tip}` / `:::{warning}` and the rest of the MyST admonition
  names become `!!! note "Optional title"` blocks with a 4-space indented body.
- `:::{dropdown} Title` becomes a collapsible `??? note "Title"` block.
- `(label)=` on the line above a heading becomes an `{#label}` attribute on that
  heading; a target that labels anything else is dropped.
- `[](#label)` cross-references get the target heading's text as their link
  text, since Python-Markdown does not fill it in.
- A table cell's code span holding an escaped pipe becomes raw
  `<code>a &#124; b</code>`. No markdown escape renders under both engines:
  Python-Markdown leaves the backslash visible inside a code span, and
  markdown-it (MyST) splits the row on a raw pipe.
- ```` ```{bibliography} ```` blocks are dropped.
- ```` ```{code-cell} ```` fences are left alone; the tutorial pipeline owns them.

Anything else that looks like a MyST directive is left untouched and reported,
so a human converts it.

Usage:
    python scripts/myst_to_zensical.py docs/index.md docs/guide/*.md
    python scripts/myst_to_zensical.py --dry-run docs/index.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Conversion", "convert", "main"]

# MyST admonition directives that map onto Python-Markdown admonition types.
ADMONITIONS = (
    "attention",
    "caution",
    "danger",
    "error",
    "hint",
    "important",
    "note",
    "seealso",
    "tip",
    "warning",
)

# Frontmatter keys Zensical understands; every other key is MyST-only.
KEPT_FRONTMATTER_KEYS = ("description", "title")

DIRECTIVE_RE = re.compile(r"^(:{3,})\{([a-z-]+)\}\s*(.*)$")
BIBLIOGRAPHY_RE = re.compile(r"^```\{bibliography\}")
LABEL_RE = re.compile(r"^\(([A-Za-z0-9_-]+)\)=\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
TOP_LEVEL_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):")
EMPTY_CROSSREF_RE = re.compile(r"\[\]\(#([A-Za-z0-9_-]+)\)")
CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
DELIMITER_CELL_RE = re.compile(r":?-+:?")
CODE_SPAN_RE = re.compile(r"`([^`]+)`")
ESCAPED_PIPE = "\\|"


@dataclass(frozen=True)
class Conversion:
    """A converted page and the MyST constructs the converter left alone."""

    text: str
    unconverted: tuple[str, ...]


def split_frontmatter(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split a page into its frontmatter lines (without the `---` fences) and body."""
    if not lines or lines[0].strip() != "---":
        return [], lines
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], lines[i + 1 :]
    return [], lines


def convert_frontmatter(frontmatter: list[str]) -> list[str]:
    """Return the frontmatter block keeping only the keys Zensical reads."""
    kept: list[str] = []
    keeping = False
    for line in frontmatter:
        key_match = TOP_LEVEL_KEY_RE.match(line)
        if key_match:
            keeping = key_match.group(1) in KEPT_FRONTMATTER_KEYS
        if keeping:
            kept.append(line)
    if not kept:
        return []
    return ["---", *kept, "---"]


def collect_labels(lines: list[str]) -> dict[str, str]:
    """Map each `(label)=` target that sits above a heading to that heading's text."""
    labels: dict[str, str] = {}
    for i, line in enumerate(lines):
        label_match = LABEL_RE.match(line)
        if not label_match:
            continue
        heading = _next_heading(lines, i + 1)
        if heading is not None:
            labels[label_match.group(1)] = heading
    return labels


def _next_heading(lines: list[str], start: int) -> str | None:
    """Return the heading text if the next non-blank line is a heading."""
    for line in lines[start:]:
        if not line.strip():
            continue
        heading_match = HEADING_RE.match(line)
        return heading_match.group(2) if heading_match else None
    return None


def convert_table_pipes(lines: list[str]) -> list[str]:
    """Rewrite table-cell code spans that hold an escaped pipe as raw `<code>`."""
    rows = table_row_indices(lines)
    return [
        CODE_SPAN_RE.sub(_code_span_to_html, line) if i in rows else line
        for i, line in enumerate(lines)
    ]


def table_row_indices(lines: list[str]) -> set[int]:
    """Return the indices of every table row, found from each table's `---|---` line.

    Leading and trailing pipes are optional in a table row, so the delimiter line
    is the only reliable marker of where a table starts.
    """
    rows: set[int] = set()
    in_fence = False
    for i, line in enumerate(lines):
        if CODE_FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not _is_delimiter_row(line):
            continue
        if i and "|" in lines[i - 1]:
            rows.add(i - 1)
        for j in range(i + 1, len(lines)):
            if not lines[j].strip() or "|" not in lines[j]:
                break
            rows.add(j)
    return rows


def _is_delimiter_row(line: str) -> bool:
    """Is this the `---|---` line that separates a table's header from its body?"""
    stripped = line.strip()
    if "|" not in stripped or set(stripped) - set("|-: "):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    return all(DELIMITER_CELL_RE.fullmatch(cell) for cell in cells)


def _code_span_to_html(match: re.Match[str]) -> str:
    """Return a code span as raw `<code>`, pipe included, or unchanged if it has none."""
    span = match.group(1)
    if ESCAPED_PIPE not in span:
        return match.group(0)
    escaped = span.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<code>{escaped.replace(ESCAPED_PIPE, '&#124;')}</code>"


def convert(text: str) -> Conversion:
    """Convert one MyST page to Zensical markdown."""
    frontmatter, body = split_frontmatter(text.split("\n"))
    labels = collect_labels(body)
    unconverted: list[str] = []
    out: list[str] = []
    # Open directives, each as (closing marker, whether its body is indented).
    stack: list[tuple[str, bool]] = []
    i = 0
    while i < len(body):
        line = body[i]
        indent = "    " * sum(1 for _, indented in stack if indented)

        if BIBLIOGRAPHY_RE.match(line):
            i = _end_of_fence(body, i + 1) + 1
            if i < len(body) and not body[i].strip():
                i += 1
            continue

        label_match = LABEL_RE.match(line)
        if label_match:
            i += 1
            if label_match.group(1) in labels:
                i = _emit_labelled_heading(body, i, label_match.group(1), indent, out)
            continue

        directive_match = DIRECTIVE_RE.match(line.strip())
        if directive_match:
            marker, name, title = directive_match.groups()
            title = title.strip()
            if name in ADMONITIONS:
                out.append(indent + f"!!! {name}" + (f' "{title}"' if title else ""))
                stack.append((marker, True))
            elif name == "dropdown":
                out.append(indent + f'??? note "{title}"')
                stack.append((marker, True))
            else:
                unconverted.append(f"{marker}{{{name}}}")
                out.append(line)
                stack.append((marker, False))
            i += 1
            continue

        if stack and line.strip() == stack[-1][0]:
            stack.pop()
            i += 1
            continue

        out.append(indent + line if line.strip() else line)
        i += 1

    converted = "\n".join(convert_table_pipes([*convert_frontmatter(frontmatter), *out]))
    converted = EMPTY_CROSSREF_RE.sub(
        lambda m: f"[{labels[m.group(1)]}](#{m.group(1)})"
        if m.group(1) in labels
        else m.group(0),
        converted,
    )
    for unresolved in EMPTY_CROSSREF_RE.findall(converted):
        unconverted.append(f"[](#{unresolved})")
    return Conversion(converted.strip("\n") + "\n", tuple(unconverted))


def _end_of_fence(lines: list[str], start: int) -> int:
    """Return the index of the closing ``` of a fence whose body starts at `start`."""
    for i in range(start, len(lines)):
        if lines[i].strip() == "```":
            return i
    return len(lines)


def _emit_labelled_heading(
    lines: list[str], start: int, label: str, indent: str, out: list[str]
) -> int:
    """Emit the heading `label` targets with an `{#label}` attribute, and return the next index."""
    i = start
    while i < len(lines) and not lines[i].strip():
        out.append(lines[i])
        i += 1
    heading_match = HEADING_RE.match(lines[i]) if i < len(lines) else None
    if heading_match is None:
        return i
    hashes, title = heading_match.groups()
    out.append(f"{indent}{hashes} {title} {{#{label}}}")
    return i + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="Pages to convert in place")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing",
    )
    args = parser.parse_args()
    for path in args.paths:
        original = path.read_text()
        result = convert(original)
        changed = result.text != original
        if changed and not args.dry_run:
            path.write_text(result.text)
        status = "unchanged" if not changed else "would convert" if args.dry_run else "converted"
        print(f"{status}: {path}")
        for construct in result.unconverted:
            print(f"  needs a hand fix: {construct}")


if __name__ == "__main__":
    main()
