"""Fail the docs build if any executed tutorial cell wrote to stderr.

Tutorial cells must not *trigger* warnings (issue #489): a warning on a docs
page means the analysis should change, not that the message should be hidden.
`docs/myst.yml` sets `output_stderr: remove-warn`, which strips stderr from the
rendered site, so this script reads MyST's *execute cache* instead — the raw
kernel outputs in `docs/_build/execute/<key>.json`, one file per page.

Each cache file is a JSON list with one entry per executed code cell, holding
that cell's Jupyter outputs (`{"output_type": "stream", "name": "stderr",
"text": ...}` is what we look for). The filename is mystmd's cache key:
``md5(kernelspec.name + JSON.stringify([{kind, content, raisesException}, ...]))``
over the page's code cells. We rebuild that key from each tutorial `.md` so
only the *current* version of every page is inspected — stale entries from
earlier builds are ignored, and a page with no entry (never executed) is an
error rather than a silent pass.

Usage (wired as the last step of `uv run poe docs-site`)::

    uv run python scripts/check_tutorial_stderr.py [--docs-dir docs]

Exit status 1 on any stderr not covered by ``ALLOWED_STDERR`` or on a missing
cache entry; 0 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# (page path relative to docs/, substring of the stderr text) pairs that are
# deliberately tolerated. Empty by design: add an entry only with a comment
# naming the lesson that needs it.
ALLOWED_STDERR: tuple[tuple[str, str], ...] = ()

TUTORIAL_GLOB = "tutorials/*/*.md"
EXECUTE_CACHE = Path("_build") / "execute"

# A MyST-NB code cell: ```{code-cell} <lang> ... ```. Directive options
# (`:tags: [...]`) sit on the first body lines and are not part of the code.
_CODE_CELL_RE = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)
_KERNEL_NAME_RE = re.compile(
    r"^kernelspec:\n(?:[ \t]+[^\n]*\n)*?[ \t]+name:[ \t]*(\S+)", re.M
)


@dataclass(frozen=True)
class StderrHit:
    page: str
    cell_index: int
    first_line: str
    text: str


def code_cells(md_text: str) -> list[str]:
    """Return the source of every `{code-cell}` block, options stripped."""
    cells = []
    for match in _CODE_CELL_RE.finditer(md_text):
        lines = match.group(1).split("\n")
        while lines and lines[0].startswith(":"):
            lines.pop(0)
        cells.append("\n".join(lines).rstrip("\n"))
    return cells


def kernel_name(md_text: str, default: str = "python3") -> str:
    """Kernel name from the page's `kernelspec` frontmatter."""
    match = _KERNEL_NAME_RE.search(md_text)
    return match.group(1) if match else default


def cache_key(cells: list[str], kernel: str) -> str:
    """Reproduce mystmd's execute-cache key for a page's code cells."""
    items = [{"kind": "block", "content": c, "raisesException": False} for c in cells]
    # JSON.stringify: compact separators, non-ASCII left literal.
    payload = json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.md5()
    digest.update(kernel.encode("utf-8"))
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def find_stderr(outputs_per_cell: list) -> list[tuple[int, str]]:
    """(cell index, text) for every stderr stream in a cache entry."""
    hits = []
    for index, outputs in enumerate(outputs_per_cell):
        if not isinstance(outputs, list):  # inline expressions store a dict
            continue
        for output in outputs:
            if output.get("output_type") == "stream" and output.get("name") == "stderr":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                hits.append((index, text))
    return hits


def check_pages(
    docs_dir: Path,
    allowlist: list[tuple[str, str]] | tuple[tuple[str, str], ...] = ALLOWED_STDERR,
) -> tuple[list[StderrHit], list[str]]:
    """Scan every tutorial page's cache entry.

    Returns ``(hits, missing)``: stderr hits not covered by ``allowlist``, and
    the pages that have no execute-cache entry at all.
    """
    docs_dir = Path(docs_dir)
    cache_dir = docs_dir / EXECUTE_CACHE
    hits: list[StderrHit] = []
    missing: list[str] = []
    for page_path in sorted(docs_dir.glob(TUTORIAL_GLOB)):
        page = page_path.relative_to(docs_dir).as_posix()
        md_text = page_path.read_text(encoding="utf-8")
        cells = code_cells(md_text)
        if not cells:
            continue
        entry = cache_dir / f"{cache_key(cells, kernel_name(md_text))}.json"
        if not entry.exists():
            missing.append(page)
            continue
        outputs_per_cell = json.loads(entry.read_text(encoding="utf-8"))
        for index, text in find_stderr(outputs_per_cell):
            if any(page == p and needle in text for p, needle in allowlist):
                continue
            first_line = cells[index].split("\n", 1)[0] if index < len(cells) else ""
            hits.append(StderrHit(page, index, first_line, text))
    return hits, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--docs-dir",
        default=Path(__file__).resolve().parents[1] / "docs",
        type=Path,
        help="docs/ directory holding tutorials/ and _build/execute/ (default: repo docs/)",
    )
    args = parser.parse_args(argv)

    hits, missing = check_pages(args.docs_dir)
    for page in missing:
        print(f"{page}: no execute-cache entry (page was not executed in this build)")
    for hit in hits:
        print(f"\n{hit.page}  code cell {hit.cell_index}  ({hit.first_line})")
        for line in hit.text.rstrip("\n").split("\n"):
            print(f"    {line}")
    if hits or missing:
        print(
            f"\n{len(hits)} stderr output(s), {len(missing)} unexecuted page(s). "
            "Tutorial cells must not emit warnings — change the analysis so the "
            "warning is not triggered (see docs/myst.yml `output_stderr`)."
        )
        return 1
    print("check_tutorial_stderr: no stderr in any executed tutorial cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
