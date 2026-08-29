#!/usr/bin/env python
"""Convert marimo tutorial notebooks to MyST-NB markdown for the docs site.

The marimo ``.py`` notebooks under ``docs/tutorials/`` are the single source of
truth for the tutorials — we never hand-author the ``.md`` files. This script
shells out to ``marimo export md`` (which turns ``mo.md()`` cells into prose and
code cells into fenced blocks), then rewrites the marimo-flavored markdown into
MyST-NB markdown that Jupyter Book v2 executes at build time to bake outputs.

It is run by the ``docs-generate`` poe task alongside the griffe2md API docs, so
building the docs regenerates the tutorial ``.md`` from the ``.py`` every time
(deterministic output → MyST's execute cache still skips unchanged cells).

Transforms:

* the marimo frontmatter (``title``/``marimo-version``/``header``…) is replaced
  with a ``kernelspec`` block so MyST executes the page through the ``python3``
  kernel;
* a "Run this tutorial locally" note linking to the source notebook on GitHub is
  inserted after the title heading;
* ```` ```python {.marimo} ```` fences become ```` ```{code-cell} python3 ````;
* ``hide_code="true"`` code cells (marimo's "hide the source, show the output")
  map to ``:tags: [remove-input]``;
* an optional first-line ``# myst: <tags>`` directive **overrides** the tag set
  (e.g. ``# myst: remove-cell`` to hide input *and* output, ``# myst:
  remove-stderr``);
* ``import marimo`` lines are stripped and any cell left empty is dropped;
* marimo ``/// name | title … ///`` fences become MyST ``:::{name} title … :::``
  colon-fences so admonitions render.

Usage::

    python scripts/marimo_to_myst.py docs/tutorials/workflows/01_glm.py
    python scripts/marimo_to_myst.py --all   # every notebook in TUTORIAL_GLOBS
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Notebooks converted by `--all` (and by the `docs-generate` poe task).
TUTORIAL_GLOBS = [
    "docs/tutorials/basics/[0-9]*.py",
    "docs/tutorials/workflows/[0-9]*.py",
]

# Where the source notebooks live on GitHub (for the "run locally" banner).
GITHUB_BLOB = "https://github.com/cosanlab/nltools/blob/master"

FRONTMATTER = """\
---
# AUTO-GENERATED from {source} by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
kernelspec:
  name: python3
  display_name: Python 3
---
"""

# A marimo code cell in the exported markdown, with optional attributes such as
# `hide_code="true"` after `.marimo`.
CELL_RE = re.compile(
    r"^```python \{\.marimo(?P<attrs>[^}]*)\}\n(?P<body>.*?)\n```$",
    re.DOTALL | re.MULTILINE,
)

# `import marimo` / `import marimo as mo` — dead once mo.md() cells are prose.
MARIMO_IMPORT_RE = re.compile(r"^\s*import\s+marimo\b.*$", re.MULTILINE)

# Optional `# myst: tag1 tag2` directive on the first line of a cell.
MYST_DIRECTIVE_RE = re.compile(r"^\s*#\s*myst:\s*(?P<tags>.+?)\s*$")

# A marimo `/// name | title … ///` fence → MyST `:::` colon-fence.
MARIMO_FENCE_OPEN_RE = re.compile(
    r"^/// (?P<name>\w+)(?: \| (?P<title>.*))?$", re.MULTILINE
)


def export_marimo_md(notebook: Path) -> str:
    """Return the marimo-flavored markdown export of a notebook."""
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "md", str(notebook)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def rel_to_repo(path: Path) -> Path:
    """Path relative to the repo root, or the bare name if it lives elsewhere."""
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return Path(path.name)


def strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited YAML frontmatter block, if present."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]


def convert_admonitions(text: str) -> str:
    """Convert marimo `///` fences to MyST `:::` colon-fences."""

    def _open(match: re.Match) -> str:
        name = match.group("name")
        title = match.group("title")
        return f":::{{{name}}} {title}" if title else f":::{{{name}}}"

    text = MARIMO_FENCE_OPEN_RE.sub(_open, text)
    # Closing bare `///` fences → `:::` (only those left after the open-rewrite).
    text = re.sub(r"^///\s*$", ":::", text, flags=re.MULTILINE)
    return text


def transform_cell(attrs: str, body: str) -> str | None:
    """Rewrite one marimo code-cell into a MyST ``{code-cell}`` block.

    Returns ``None`` for cells that are empty after stripping marimo imports
    (caller drops them).
    """
    lines = body.split("\n")

    # An explicit `# myst:` directive on the first line overrides everything.
    tags: list[str] = []
    if lines:
        match = MYST_DIRECTIVE_RE.match(lines[0])
        if match:
            tags = match.group("tags").replace(",", " ").split()
            lines = lines[1:]
    # Otherwise, marimo's hide_code (hide source, show output) → remove-input.
    if not tags and 'hide_code="true"' in attrs:
        tags = ["remove-input"]

    code = MARIMO_IMPORT_RE.sub("", "\n".join(lines)).strip("\n")
    if not code.strip():
        return None

    header = "```{code-cell} python3"
    if tags:
        header += "\n:tags: [" + ", ".join(tags) + "]"
    return f"{header}\n{code}\n```"


def source_banner(notebook: Path) -> str:
    """A MyST admonition pointing at the marimo notebook this page is rendered from."""
    rel = rel_to_repo(notebook).as_posix()
    return (
        ":::{tip} Run this tutorial locally\n"
        "The outputs below were baked in at build time. This page is rendered from a "
        f"[marimo]({'https://marimo.io'}) notebook — "
        f"[`{rel}`]({GITHUB_BLOB}/{rel}) — that you can open and edit locally with "
        f"`uvx marimo edit --sandbox {Path(rel).name}`.\n"
        ":::\n"
    )


def insert_banner(body: str, banner: str) -> str:
    """Insert the banner just after the first top-level ``# `` heading.

    Skips ``# `` lines inside fenced code blocks (Python comments), which are not
    Markdown headings.
    """
    lines = body.split("\n")
    in_fence = False
    for i, line in enumerate(lines):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and line.startswith("# "):
            rest = "\n".join(lines[i + 1 :]).lstrip("\n")
            head = "\n".join(lines[: i + 1])
            return f"{head}\n\n{banner}\n{rest}"
    # No heading found → prepend.
    return f"{banner}\n{body}"


def convert(notebook: Path) -> Path:
    """Convert one marimo notebook to a sibling ``.md`` and return its path."""
    raw = export_marimo_md(notebook)
    body = strip_frontmatter(raw).lstrip("\n")
    body = convert_admonitions(body)
    body = insert_banner(body, source_banner(notebook))

    def _replace(match: re.Match) -> str:
        out = transform_cell(match.group("attrs"), match.group("body"))
        # Sentinel marks empty cells for cleanup of their surrounding blank lines.
        return out if out is not None else "\x00DROP\x00"

    body = CELL_RE.sub(_replace, body)
    body = re.sub(r"\n*\x00DROP\x00\n*", "\n\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")

    rel = rel_to_repo(notebook)
    out_text = FRONTMATTER.format(source=rel.name) + "\n" + body + "\n"
    out_path = notebook.with_suffix(".md")
    out_path.write_text(out_text)
    return out_path


def resolve_targets(args: argparse.Namespace) -> list[Path]:
    if args.all:
        targets: list[Path] = []
        for pattern in TUTORIAL_GLOBS:
            targets.extend(sorted(REPO_ROOT.glob(pattern)))
        return targets
    return [Path(p).resolve() for p in args.notebooks]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("notebooks", nargs="*", help="marimo .py notebooks to convert")
    parser.add_argument(
        "--all", action="store_true", help="convert every notebook in TUTORIAL_GLOBS"
    )
    args = parser.parse_args()

    targets = resolve_targets(args)
    if not targets:
        parser.error("no notebooks given (pass paths or --all)")

    for nb in targets:
        if not nb.exists():
            print(f"  skip (missing): {nb}", file=sys.stderr)
            continue
        out = convert(nb)
        print(f"  {rel_to_repo(nb)} -> {rel_to_repo(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
