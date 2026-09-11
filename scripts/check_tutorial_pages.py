#!/usr/bin/env python
"""Assert that every built tutorial page carries the outputs its cells produced.

The tutorials render through `scripts/docs_show.py`, which the generated pages
activate by importing it — which only works because the docs tasks put
``scripts/`` on ``PYTHONPATH``. When that import fails, markdown-exec catches the
error, renders the traceback into the page, and falls back to its own ``python``
formatter for every later cell: no figures, no reprs, no stderr gate. Zensical
reports no issue and the build exits 0 with seven gutted pages.

So the build checks the result. Every notebook under ``docs/tutorials/`` must
have produced a page holding at least one ``cell-output`` block; a page with none
fails the task and is named.

Usage::

    python scripts/check_tutorial_pages.py              # checks ./site
    python scripts/check_tutorial_pages.py --site-dir build
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The notebooks whose pages must carry outputs. Kept in step with
# scripts/marimo_to_zensical.py's TUTORIAL_GLOBS.
TUTORIAL_GLOBS = [
    "docs/tutorials/basics/[0-9]*.py",
    "docs/tutorials/data-operations/[0-9]*.py",
    "docs/tutorials/analysis/[0-9]*.py",
    "docs/tutorials/workflows/[0-9]*.py",
]

# What scripts/docs_show.py wraps every printed value, repr and figure in.
OUTPUT_MARKER = 'class="cell-output'


def notebooks() -> list[Path]:
    """Every tutorial notebook, in page order."""
    found: list[Path] = []
    for pattern in TUTORIAL_GLOBS:
        found.extend(sorted(REPO_ROOT.glob(pattern)))
    return found


def page_of(notebook: Path, site_dir: Path) -> Path:
    """The built page for `notebook`, under zensical's directory URLs."""
    group = notebook.parent.name
    return site_dir / "tutorials" / group / notebook.stem / "index.html"


def check(site_dir: Path) -> list[str]:
    """Return one complaint per tutorial page that is missing or has no output."""
    problems = []
    for notebook in notebooks():
        page = page_of(notebook, site_dir)
        if not page.exists():
            problems.append(f"{page}: not built from {notebook.name}")
        elif OUTPUT_MARKER not in page.read_text(errors="replace"):
            problems.append(
                f"{page}: no cell output — {notebook.name} rendered without "
                "running through scripts/docs_show.py"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--site-dir", default="site", help="built site to check (default: site)"
    )
    args = parser.parse_args()

    site_dir = Path(args.site_dir)
    if not site_dir.is_absolute():
        site_dir = REPO_ROOT / site_dir

    problems = check(site_dir)
    if problems:
        print(
            "check_tutorial_pages: tutorial pages lost their outputs", file=sys.stderr
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(
        f"check_tutorial_pages: {len(notebooks())} tutorial pages carry their outputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
