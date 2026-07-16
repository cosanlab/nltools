#!/usr/bin/env python
"""Convert marimo notebooks to MyST-NB markdown for the docs site.

The marimo ``.py`` notebooks under ``docs/tutorials/`` are the single source of
truth for the tutorials. We never hand-author the ``.md`` files; this script
generates them.

It shells out to ``marimo export md`` (which turns ``mo.md()`` cells into prose
and code cells into fenced blocks), then rewrites the marimo-flavored markdown
into MyST-NB markdown that the Jupyter Book v2 site executes at build time:

* the marimo frontmatter (``title``/``marimo-version``/``width``/``header``...)
  is replaced with a standard ``kernelspec`` block (mystmd executes ``{code-cell}``
  directives at build time; the old MyST-NB ``file_format`` key is ignored by
  mystmd v2, so it is intentionally omitted);
* ```` ```python {.marimo} ```` fences become ```` ```{code-cell} python3 ````;
* ``import marimo`` lines are stripped, and any cell that becomes empty as a
  result is dropped entirely;
* an optional first-line ``# myst: <tags>`` directive inside a cell is lifted
  onto the code-cell as ``:tags: [<tags>]`` (e.g. ``# myst: remove-input`` hides
  the source, ``# myst: remove-cell`` hides input + output). This is the one
  hook for controlling cell visibility in the rendered docs while keeping the
  source as plain marimo Python.

Usage::

    python scripts/marimo_to_myst.py docs/tutorials/workflows/01_glm.py
    python scripts/marimo_to_myst.py docs/tutorials/workflows/*.py
    python scripts/marimo_to_myst.py --all   # every notebook in TUTORIAL_GLOBS
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def rel_to_repo(path: Path) -> Path:
    """Path relative to the repo root, or the bare name if it lives elsewhere."""
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return Path(path.name)

# Notebooks converted by `--all` (and by the `tutorials-build` poe task).
TUTORIAL_GLOBS = [
    "docs/tutorials/workflows/[0-9]*.py",
]

FRONTMATTER = """\
---
# AUTO-GENERATED from {source} by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe tutorials-build`.
kernelspec:
  name: python3
  display_name: Python 3
---
"""

# A marimo code cell in the exported markdown.
CELL_RE = re.compile(r"^```python \{\.marimo\}\n(?P<body>.*?)\n```$", re.DOTALL | re.MULTILINE)

# `import marimo` / `import marimo as mo` — dead once mo.md() cells are prose.
MARIMO_IMPORT_RE = re.compile(r"^\s*import\s+marimo\b.*$", re.MULTILINE)

# Optional `# myst: tag1 tag2` directive on the first line of a cell.
MYST_DIRECTIVE_RE = re.compile(r"^\s*#\s*myst:\s*(?P<tags>.+?)\s*$")


def export_marimo_md(notebook: Path) -> str:
    """Return the marimo-flavored markdown export of a notebook."""
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "md", str(notebook)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited YAML frontmatter block, if present."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]


def transform_cell(body: str) -> str | None:
    """Rewrite one marimo code-cell body into a MyST ``{code-cell}`` block.

    Returns ``None`` for cells that are empty after stripping marimo imports
    (caller drops them).
    """
    lines = body.split("\n")

    tags: list[str] = []
    if lines:
        match = MYST_DIRECTIVE_RE.match(lines[0])
        if match:
            tags = match.group("tags").replace(",", " ").split()
            lines = lines[1:]

    code = MARIMO_IMPORT_RE.sub("", "\n".join(lines)).strip("\n")
    if not code.strip():
        return None

    header = "```{code-cell} python3"
    if tags:
        header += "\n:tags: [" + ", ".join(tags) + "]"
    return f"{header}\n{code}\n```"


def convert(notebook: Path) -> Path:
    """Convert one marimo notebook to a sibling ``.md`` and return its path."""
    raw = export_marimo_md(notebook)
    body = strip_frontmatter(raw).lstrip("\n")

    rel = rel_to_repo(notebook)

    def _replace(match: re.Match) -> str:
        out = transform_cell(match.group("body"))
        # Sentinel marks empty cells for cleanup of their surrounding blank lines.
        return out if out is not None else "\x00DROP\x00"

    body = CELL_RE.sub(_replace, body)
    body = re.sub(r"\n*\x00DROP\x00\n*", "\n\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")

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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", help="marimo .py notebooks to convert")
    parser.add_argument("--all", action="store_true", help="convert every notebook in TUTORIAL_GLOBS")
    args = parser.parse_args()

    targets = resolve_targets(args)
    if not targets:
        parser.error("no notebooks given (pass paths or --all)")

    for nb in targets:
        if not nb.exists():
            print(f"  skip (missing): {nb}", file=sys.stderr)
            continue
        out = convert(nb)
        print(f"  {rel_to_repo(nb)} -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
