#!/usr/bin/env python
"""Convert marimo tutorial notebooks to MyST-NB markdown for the docs site.

The marimo ``.py`` notebooks under ``docs/tutorials/`` are the single source of
truth for the tutorials — we never hand-author the ``.md`` files. This script
shells out to ``marimo export md`` (which turns ``mo.md()`` cells into prose and
code cells into fenced blocks), then rewrites the marimo-flavored markdown into
MyST-NB markdown that Jupyter Book v2 executes at build time to bake outputs.

It is run by the ``docs-generate`` poe task alongside the vocabulary tables, so
building the docs regenerates the tutorial ``.md`` from the ``.py`` every time
(deterministic output → MyST's execute cache still skips unchanged cells).

Transforms:

* the marimo frontmatter (``title``/``marimo-version``/``header``…) is replaced
  with a ``kernelspec`` block so MyST executes the page through the ``python3``
  kernel, plus ``edit_url``/``source_url``/``downloads`` that point the theme's
  edit, source, and download links at the ``.py`` notebook (and at molab)
  rather than at this generated ``.md``;
* an "Open in molab" badge and a "Run this tutorial" tip (cloud via molab, or
  locally via ``uvx marimo edit --sandbox``) are inserted after the title heading;
* ```` ```python {.marimo} ```` fences become ```` ```{code-cell} python3 ````;
* ``hide_code="true"`` code cells (marimo's "hide the source, show the output")
  map to ``:tags: [remove-input]``;
* an optional first-line ``# myst: <tags>`` directive **overrides** the tag set
  (e.g. ``# myst: remove-cell`` to hide input *and* output, ``# myst:
  remove-stderr``);
* ``import marimo`` lines are stripped and any cell left empty is dropped;
* marimo ``/// name | title … ///`` fences become MyST ```` ````{name} title
  … ```` ```` backtick-fences so admonitions render. Backticks, not colons: the
  generated pages sit inside the zensical site's ``docs_dir``, where a line
  starting ``:::`` is an mkdocstrings directive and fails the build.

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

FRONTMATTER = """\
---
# AUTO-GENERATED from {source} by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
kernelspec:
  name: python3
  display_name: Python 3
edit_url: {edit_url}
source_url: {source_url}
downloads:
  - url: {molab_url}
    title: Open in molab
  - url: {source_url}
    title: Source notebook ({name})
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

# A marimo `/// name | title … ///` fence → MyST backtick directive fence.
MARIMO_FENCE_OPEN_RE = re.compile(
    r"^/// (?P<name>\w+)(?: \| (?P<title>.*))?$", re.MULTILINE
)

# Four backticks so a directive body can still hold an ordinary ``` code fence.
DIRECTIVE_FENCE = "````"


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


def github_url(rel: str) -> str:
    """GitHub blob URL of a repo-relative posix path on the deploy branch."""
    return f"{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel}"


def github_edit_url(rel: str) -> str:
    """GitHub edit URL of a repo-relative posix path on the deploy branch."""
    return f"{GITHUB_REPO}/edit/{GITHUB_BRANCH}/{rel}"


def molab_url(rel: str) -> str:
    """molab URL that opens the GitHub-hosted notebook at ``rel``."""
    return f"{MOLAB_GITHUB}/{GITHUB_OWNER_REPO}/blob/{GITHUB_BRANCH}/{rel}"


def molab_badge(rel: str) -> str:
    """marimo's official "Open in molab" badge, linked to the notebook at ``rel``."""
    return f"[![Open in molab]({MOLAB_SHIELD})]({molab_url(rel)})"


def frontmatter(rel: str) -> str:
    """MyST page frontmatter for the page rendered from the notebook at ``rel``."""
    return FRONTMATTER.format(
        source=rel,
        name=Path(rel).name,
        edit_url=github_edit_url(rel),
        source_url=github_url(rel),
        molab_url=molab_url(rel),
    )


def strip_frontmatter(text: str) -> str:
    """Drop a leading ``---``-delimited YAML frontmatter block, if present."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :]


def convert_admonitions(text: str) -> str:
    """Convert marimo `///` fences to MyST backtick directive fences."""

    def _open(match: re.Match) -> str:
        name = match.group("name")
        title = match.group("title")
        head = f"{DIRECTIVE_FENCE}{{{name}}}"
        return f"{head} {title}" if title else head

    text = MARIMO_FENCE_OPEN_RE.sub(_open, text)
    # Closing bare `///` fences (only those left after the open-rewrite).
    text = re.sub(r"^///\s*$", DIRECTIVE_FENCE, text, flags=re.MULTILINE)
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


def source_banner(rel: str) -> str:
    """The molab badge plus a tip on running the notebook at ``rel`` yourself."""
    name = Path(rel).name
    return (
        f"{molab_badge(rel)}\n"
        "\n"
        f"{DIRECTIVE_FENCE}{{tip}} Run this tutorial\n"
        "This page is rendered from the [marimo](https://marimo.io) notebook "
        f"[`{rel}`]({github_url(rel)}). Click the badge to run it in the cloud (free, "
        f"no install), or locally: download `{name}` and run "
        f"`uvx marimo edit --sandbox {name}`. Outputs below were baked in at build time.\n"
        f"{DIRECTIVE_FENCE}\n"
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
    rel = rel_to_repo(notebook).as_posix()
    raw = export_marimo_md(notebook)
    body = strip_frontmatter(raw).lstrip("\n")
    body = convert_admonitions(body)
    body = insert_banner(body, source_banner(rel))

    def _replace(match: re.Match) -> str:
        out = transform_cell(match.group("attrs"), match.group("body"))
        # Sentinel marks empty cells for cleanup of their surrounding blank lines.
        return out if out is not None else "\x00DROP\x00"

    body = CELL_RE.sub(_replace, body)
    body = re.sub(r"\n*\x00DROP\x00\n*", "\n\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")

    out_text = frontmatter(rel) + "\n" + body + "\n"
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
