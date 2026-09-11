#!/usr/bin/env python3
"""Enforce the Google-style Markdown docstring conventions (CLAUDE.md).

Docstrings are rendered by mkdocstrings-python. Constructs that griffe's Google
parser does not understand render as literal text on the live site, so this
check fails on them at lint time:

- ``section-header``: a singular or off-style section header (``Example:``,
  ``Return:``, ``Arg:``, ``Parameter(s):``, ``Raise:``) — griffe renders these
  as plain admonitions, so a doctest under ``Example:`` becomes prose.
- ``doctest``: a ``>>>`` prompt line. House style is a fenced ```` ```python ````
  block under ``Examples:`` (prompts dropped; expected output as ``# → ...``).
- ``rst``: RST leftovers — ``:param``, ``:returns:``, ``:rtype:``, roles like
  ``:class:`` / ``:func:`` / ``:meth:``, ``.. directive::`` blocks, and a line
  ending in ``::`` (RST literal-block marker).
- ``legacy-arg``: an Args/Attributes entry written ``name: (type) desc``
  instead of ``name (type): desc``.
- ``summary``: the first docstring line does not end in ``.``/``?``/``!`` or
  exceeds 120 characters. griffe uses that physical line as the one-line
  summary in member tables, so a summary that wraps mid-phrase is truncated.

Scope: every module, class, function, and attribute docstring in the
``nltools`` package (tests excluded). Fenced code blocks inside a docstring are
skipped for the header/RST/legacy-arg rules (a ``::`` inside code is code).

Usage:  python scripts/check_docstrings.py [PATH ...]
Exit status 1 if any finding exists (so it can gate CI).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple
from collections.abc import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import iter_py_files, rel_posix  # noqa: E402

DEFAULT_ROOTS = ["nltools"]
EXCLUDE_PARTS = {"tests"}

CATEGORIES = ("section-header", "doctest", "rst", "legacy-arg", "summary")
SUMMARY_MAX_LEN = 120

# Headers griffe does not recognise (or that are off the house style), lower-case,
# without the trailing colon.
BAD_HEADERS = frozenset(
    {
        "example",
        "return",
        "arg",
        "parameter",
        "parameters",
        "raise",
        "yield",
        "attribute",
    }
)
# Sections whose entries follow the ``name (type): description`` form.
ARG_SECTIONS = frozenset(
    {
        "args",
        "arguments",
        "keyword args",
        "keyword arguments",
        "other args",
        "attributes",
    }
)

_HEADER = re.compile(r"^(?P<indent>\s*)(?P<title>[A-Za-z][A-Za-z ]*):\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_LEGACY_ARG = re.compile(r"^\s+\*{0,2}\w+:\s*\(")
_RST = [
    re.compile(r":param\b"),
    re.compile(r":returns?:"),
    re.compile(r":rtype:"),
    re.compile(r":(class|func|meth|mod|attr|obj|ref|exc|data|py:\w+):`"),
    re.compile(r"^\s*\.\. \w[\w-]*::"),
    re.compile(r"::\s*$"),
]


class Finding(NamedTuple):
    path: str
    lineno: int
    category: str
    message: str


def iter_docstrings(tree: ast.Module) -> Iterator[ast.Constant]:
    """Yield the string node of every module/class/function/attribute docstring.

    Attribute docstrings are the bare string statements that directly follow an
    assignment in a module or class body (griffe reads them as the attribute's
    documentation).
    """
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if body and _is_str_expr(body[0]):
            yield body[0].value
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for prev, stmt in zip(body, body[1:]):
                if isinstance(prev, (ast.Assign, ast.AnnAssign)) and _is_str_expr(stmt):
                    yield stmt.value


def _is_str_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def check_docstring(doc: ast.Constant, rel: str) -> list[Finding]:
    """Apply every rule to one docstring node; line numbers are absolute."""
    text: str = doc.value
    lines = text.split("\n")
    findings: list[Finding] = []

    def add(offset: int, category: str, message: str) -> None:
        findings.append(Finding(rel, doc.lineno + offset, category, message))

    summary_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if summary_idx is not None:
        summary = lines[summary_idx].strip()
        if len(summary) > SUMMARY_MAX_LEN:
            add(summary_idx, "summary", f"summary line is {len(summary)} chars (>120)")
        elif summary.rstrip(")\"'`") and summary.rstrip(")\"'`")[-1] not in ".?!":
            add(
                summary_idx,
                "summary",
                f"summary line does not end in ./?/!: {summary!r}",
            )

    in_fence = False
    arg_section_indent: int | None = None  # indent of the current Args-like header
    entry_indent: int | None = None  # indent of that section's entries
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if ">>>" in line and line.lstrip().startswith(">>>"):
            add(i, "doctest", "`>>>` doctest prompt; use a fenced ```python block")
        if in_fence:
            continue

        stripped = line.strip()
        if arg_section_indent is not None and stripped:
            if _indent(line) <= arg_section_indent:
                arg_section_indent = entry_indent = None
            else:
                # The first body line fixes the entry indentation; deeper lines
                # are description continuations and never entries.
                entry_indent = entry_indent or _indent(line)
                if _indent(line) == entry_indent and _LEGACY_ARG.match(line):
                    add(
                        i,
                        "legacy-arg",
                        f"legacy `name: (type)` entry; use `name (type):`: {stripped!r}",
                    )

        header = _HEADER.match(line)
        if header:
            title = header.group("title").strip().lower()
            if title in BAD_HEADERS:
                add(i, "section-header", f"`{stripped}` is not a griffe section header")
            if title in ARG_SECTIONS:
                arg_section_indent = len(header.group("indent"))
                entry_indent = None

        for pattern in _RST:
            if pattern.search(line):
                add(i, "rst", f"RST syntax: {stripped!r}")
                break
    return findings


def check_source(source: str, rel: str) -> list[Finding]:
    """Return every finding in one file's source text (``rel`` labels them)."""
    tree = ast.parse(source, filename=rel)
    findings: list[Finding] = []
    for doc in iter_docstrings(tree):
        findings.extend(check_docstring(doc, rel))
    return sorted(findings, key=lambda f: (f.lineno, f.category))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", default=DEFAULT_ROOTS)
    args = parser.parse_args(argv[1:])

    findings: list[Finding] = []
    for path in iter_py_files(args.paths, exclude_parts=EXCLUDE_PARTS):
        try:
            source = path.read_text()
            findings.extend(check_source(source, rel_posix(path)))
        except SyntaxError as e:  # pragma: no cover - surfaced to caller
            print(f"{rel_posix(path)}: SyntaxError: {e}", file=sys.stderr)

    for f in findings:
        print(f"{f.path}:{f.lineno}: [{f.category}] {f.message}")

    counts = Counter(f.category for f in findings)
    if findings:
        print()
    for category in CATEGORIES:
        print(f"  {category}: {counts[category]}")
    print(f"check_docstrings: {len(findings)} violation(s).")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
