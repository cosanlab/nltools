#!/usr/bin/env python
"""Convert marimo tutorial notebooks to zensical pages executed by markdown-exec.

The marimo ``.py`` notebooks under ``docs/tutorials/`` are the single source of
truth for the tutorials — the ``.md`` pages are build artifacts and are never
hand-authored or committed. This script shells out to ``marimo export md``
(which turns ``mo.md()`` cells into prose and code cells into fenced blocks),
then rewrites the marimo-flavored markdown into a zensical page whose code
fences ``markdown-exec`` executes while the site builds.

It runs from the ``docs-generate`` poe task, so building the docs regenerates
every page from its notebook first.

Transforms:

* the marimo frontmatter (``marimo-version``/``header``/``width``…) is replaced
  with the page ``title``, taken from the notebook's own top-level heading;
* an "Open in molab" badge and a "Run this tutorial" tip (cloud via molab, or
  locally via ``uvx marimo edit --sandbox``) are inserted after that heading;
* a hidden first cell activates ``scripts/docs_show.py``, the formatter that
  gives the rest of the page notebook-style outputs;
* ```` ```python {.marimo} ```` fences become
  ```` ```python exec="on" session="<page-slug>" source="above" ````;
* a cell marked ``hide_code`` in marimo drops the ``source`` option, so the page
  shows its output only;
* a cell whose first line is ``# docs: hide`` gets ``render="off"``, so it still
  runs — its side effects and any warning it triggers still count — but nothing
  of it reaches the page, which is how a notebook does setup a reader does not
  need to see;
* ``import marimo`` lines are stripped and any cell left empty is dropped;
* marimo ``/// name | title … ///`` fences become ``!!! name "title"``
  admonitions with an indented body.

Usage::

    python scripts/marimo_to_zensical.py docs/tutorials/workflows/01_glm.py
    python scripts/marimo_to_zensical.py --all   # every notebook in TUTORIAL_GLOBS
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from textwrap import indent

REPO_ROOT = Path(__file__).resolve().parent.parent

# Notebooks converted by `--all` (and by the `docs-generate` poe task).
TUTORIAL_GLOBS = [
    "docs/tutorials/basics/[0-9]*.py",
    "docs/tutorials/data-operations/[0-9]*.py",
    "docs/tutorials/analysis/[0-9]*.py",
    "docs/tutorials/workflows/[0-9]*.py",
]

# Where the source notebooks live on GitHub. The docs site deploys from `master`,
# so links (and molab, which fetches the notebook from GitHub) target that branch.
GITHUB_OWNER_REPO = "cosanlab/nltools"
GITHUB_BRANCH = "master"
GITHUB_REPO = f"https://github.com/{GITHUB_OWNER_REPO}"

# molab (https://docs.marimo.io/guides/molab/) opens a GitHub-hosted notebook at
# `https://molab.marimo.io/github/{owner}/{repo}/blob/{branch}/{path}` and shows
# a static preview without login; the shield is marimo's official badge image.
MOLAB_GITHUB = "https://molab.marimo.io/github"
MOLAB_SHIELD = "https://marimo.io/molab-shield.svg"

# The formatter that turns markdown-exec output into notebook output. Importable
# because the docs tasks put `scripts/` on PYTHONPATH.
FORMATTER_MODULE = "docs_show"

# A cell that runs without appearing: the page's injected first cell, and any
# cell the notebook marks `# docs: hide`.
HIDDEN_OPTIONS = 'exec="on" session="{slug}" render="off"'

FRONTMATTER = """\
---
# AUTO-GENERATED from {source} by scripts/marimo_to_zensical.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
title: {title}
---
"""

# A marimo code cell in the exported markdown, with optional attributes such as
# `hide_code="true"` after `.marimo`. The exporter lengthens the fence when the
# cell body contains one, so the closing fence has to match the opening one.
CELL_RE = re.compile(
    r"^(?P<fence>`{3,})python \{\.marimo(?P<attrs>[^}]*)\}\n"
    r"(?P<body>.*?)\n(?P=fence)$",
    re.DOTALL | re.MULTILINE,
)

# `import marimo` / `import marimo as mo` — dead once mo.md() cells are prose.
MARIMO_IMPORT_RE = re.compile(r"^\s*import\s+marimo\b.*$", re.MULTILINE)

# `# docs: hide` on a cell's first line: run the cell, render nothing.
HIDE_DIRECTIVE_RE = re.compile(r"^\s*#\s*docs:\s*hide\s*$")

# A marimo `/// name | title … ///` admonition fence.
MARIMO_FENCE_OPEN_RE = re.compile(r"^/// (?P<name>\w+)(?: \| (?P<title>.*))?$")
MARIMO_FENCE_CLOSE_RE = re.compile(r"^///\s*$")


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


def page_slug(rel: str) -> str:
    """markdown-exec session name for the page generated from the notebook at `rel`.

    One session per page: cells on a page share globals the way notebook cells
    do, and two pages never see each other's state.
    """
    path = Path(rel)
    return f"{path.parent.name}-{path.stem}".replace("_", "-")


def github_url(rel: str) -> str:
    """GitHub blob URL of a repo-relative posix path on the deploy branch."""
    return f"{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel}"


def molab_url(rel: str) -> str:
    """molab URL that opens the GitHub-hosted notebook at `rel`."""
    return f"{MOLAB_GITHUB}/{GITHUB_OWNER_REPO}/blob/{GITHUB_BRANCH}/{rel}"


def molab_badge(rel: str) -> str:
    """marimo's official "Open in molab" badge, linked to the notebook at `rel`."""
    return f"[![Open in molab]({MOLAB_SHIELD})]({molab_url(rel)})"


def page_title(body: str) -> str:
    """The notebook's top-level heading, which titles the page and its nav entry."""
    for line in body.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def frontmatter(rel: str, title: str) -> str:
    """Page frontmatter for the page rendered from the notebook at `rel`."""
    return FRONTMATTER.format(source=rel, title=title)


def strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited YAML frontmatter block, if present."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]


def code_fence(code: str) -> str:
    """The shortest backtick fence that can hold `code`."""
    fence = "```"
    while fence in code:
        fence += "`"
    return fence


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines in prose, leaving code fences untouched.

    Dropping a cell and injecting the banner both leave blank-line runs behind.
    A cell's own blank lines are part of the source the page shows "exactly as
    written", so the collapse stops at a fence.
    """
    out: list[str] = []
    fence = ""
    blanks = 0
    for line in text.split("\n"):
        if fence:
            out.append(line)
            if line.rstrip() == fence:
                fence = ""
            continue
        if line.startswith("```"):
            fence = line[: len(line) - len(line.lstrip("`"))]
            blanks = 0
        elif not line.strip():
            blanks += 1
            if blanks > 1:
                continue
        else:
            blanks = 0
        out.append(line)
    return "\n".join(out)


def convert_admonitions(text: str) -> str:
    """Convert marimo `/// name | title … ///` fences to `!!!` admonitions."""
    out: list[str] = []
    body: list[str] | None = None
    for line in text.split("\n"):
        if body is None:
            match = MARIMO_FENCE_OPEN_RE.match(line)
            if match:
                title = match.group("title")
                head = f"!!! {match.group('name')}"
                out.append(f'{head} "{title}"' if title else head)
                body = []
                continue
            out.append(line)
        elif MARIMO_FENCE_CLOSE_RE.match(line):
            out.append(indent("\n".join(body).strip("\n"), "    "))
            body = None
        else:
            body.append(line)
    if body is not None:
        out.append(indent("\n".join(body).strip("\n"), "    "))
    return "\n".join(out)


def install_cell(rel: str, slug: str) -> str:
    """The page's first cell: hidden, and it activates the output formatter."""
    code = f'import {FORMATTER_MODULE}\n\n{FORMATTER_MODULE}.install("{rel}")'
    return f"```python {HIDDEN_OPTIONS.format(slug=slug)}\n{code}\n```"


def transform_cell(attrs: str, body: str, slug: str) -> str | None:
    """Rewrite one marimo code cell into a markdown-exec fence.

    Returns `None` for cells that are empty after stripping marimo imports
    (caller drops them).
    """
    lines = body.split("\n")
    hidden = bool(lines) and bool(HIDE_DIRECTIVE_RE.match(lines[0]))
    if hidden:
        lines = lines[1:]

    code = MARIMO_IMPORT_RE.sub("", "\n".join(lines)).strip("\n")
    if not code.strip():
        return None

    if hidden:
        # `render="off"` is read by scripts/docs_show.py: run the cell, put
        # nothing on the page. Without it markdown-exec would show the output.
        options = HIDDEN_OPTIONS.format(slug=slug)
    elif 'hide_code="true"' in attrs:
        # marimo's hide_code means "hide the source, show the output", which is
        # markdown-exec's default.
        options = f'exec="on" session="{slug}"'
    else:
        options = f'exec="on" session="{slug}" source="above"'
    fence = code_fence(code)
    return f"{fence}python {options}\n{code}\n{fence}"


def source_banner(rel: str) -> str:
    """The molab badge plus a tip on running the notebook at `rel` yourself."""
    name = Path(rel).name
    return (
        f"{molab_badge(rel)}\n"
        "\n"
        '!!! tip "Run this tutorial"\n'
        "    This page is rendered from the [marimo](https://marimo.io) notebook "
        f"[`{rel}`]({github_url(rel)}). Click the badge to run it in the cloud "
        f"(free, no install), or locally: download `{name}` and run "
        f"`uvx marimo edit --sandbox {name}`. The outputs below were produced "
        "when this page was built.\n"
    )


def insert_after_heading(body: str, block: str) -> str:
    """Insert `block` just after the first top-level ``# `` heading.

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
            return f"{head}\n\n{block}\n{rest}"
    # No heading found → prepend.
    return f"{block}\n{body}"


def convert(notebook: Path) -> Path:
    """Convert one marimo notebook to a sibling ``.md`` page and return its path."""
    rel = rel_to_repo(notebook).as_posix()
    slug = page_slug(rel)
    raw = export_marimo_md(notebook)
    body = strip_frontmatter(raw).lstrip("\n")
    body = convert_admonitions(body)
    title = page_title(body)
    body = insert_after_heading(
        body, f"{source_banner(rel)}\n{install_cell(rel, slug)}\n"
    )

    def _replace(match: re.Match) -> str:
        out = transform_cell(match.group("attrs"), match.group("body"), slug)
        # Sentinel marks empty cells for cleanup of their surrounding blank lines.
        return out if out is not None else "\x00DROP\x00"

    body = CELL_RE.sub(_replace, body)
    body = re.sub(r"\n*\x00DROP\x00\n*", "\n\n", body)
    body = collapse_blank_lines(body).strip("\n")

    out_path = notebook.with_suffix(".md")
    out_path.write_text(frontmatter(rel, title) + "\n" + body + "\n")
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
