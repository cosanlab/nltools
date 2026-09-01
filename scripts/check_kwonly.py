#!/usr/bin/env python3
"""Enforce the v0.6.0 keyword-only ``*`` marker convention (CLAUDE.md).

The convention: a keyword-only ``*`` marker is required after the primary data
argument in ``__init__`` and in any public method/function with 3+ kwargs.
This is a *structural-absence* check (the ``*`` is missing) which semgrep
patterns cannot express cleanly, so it lives here alongside ``.semgrep/rules.yml``
and is run by the same ``lint-api`` poe task.

Heuristic (matches the audit's api-consistency findings): for every public
(non-underscore) ``def``, count the positional-or-keyword parameters that carry
a default and appear *before* any ``*`` / ``*args`` marker — these are "loose
kwargs" a caller could pass positionally. If 3+ such loose kwargs exist with no
keyword-only marker separating them, flag it. ``self``/``cls`` and no-default
positional data args (e.g. ``fit(X, y)``) are not counted, so sklearn-style
``fit(X, y)`` signatures are not flagged.

Scope: the entire ``nltools`` package (tests excluded). The convention binds
uniformly — facades and algorithm-layer engines alike — because a missing ``*``
is how a parameter insertion silently shifts an argument at a dispatch site
regardless of which layer it lives in. The only exceptions are the explicit
``enforcement.kwonly_exemptions:`` entries in ``docs/_data/api-vocabulary.yml``
(the single suppression home for all lint-api carve-outs), each with a reason.

Usage:  python scripts/check_kwonly.py [PATH ...]
Exit status 1 if any violations are found (so it can gate CI).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import VOCAB_YML, iter_py_files, load_vocab, rel_posix  # noqa: E402

# Minimum number of loose (defaulted, non-keyword-only) params to require a `*`.
THRESHOLD = 3

# The whole package: the convention is uniform (no per-layer carve-outs).
DEFAULT_ROOTS = ["nltools"]

EXCLUDE_PARTS = {"tests"}


def load_exemptions(vocab_yml: Path = VOCAB_YML) -> dict[tuple[str, str], str]:
    """Load the ``*``-marker carve-outs from the vocabulary manifest.

    Suppressions live only in ``docs/_data/api-vocabulary.yml``
    (``enforcement.kwonly_exemptions:``), never inline — the manifest is the
    single home for every lint-api carve-out. Each entry names the defining
    module ``path`` (prefix-matched, like the vocabulary checker's), the
    ``function``, and a required ``reason``.

    Returns:
        Mapping of ``(path, function)`` to the documented reason.
    """
    entries = load_vocab(vocab_yml)["enforcement"].get("kwonly_exemptions", [])
    exempt: dict[tuple[str, str], str] = {}
    for entry in entries:
        try:
            exempt[(entry["path"], entry["function"])] = entry["reason"]
        except KeyError as missing:
            raise SystemExit(
                f"error: kwonly_exemptions entry {entry!r} is missing the "
                f"required {missing} field (path, function, reason)."
            ) from None
    return exempt


def _is_exempt(exempt: dict[tuple[str, str], str], rel: str, name: str) -> bool:
    """True if a manifest exemption covers this (module path, function)."""
    return any(name == fn and rel.startswith(prefix) for prefix, fn in exempt)


def loose_kwargs(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Count defaulted positional-or-keyword params before any ``*`` marker.

    Excludes ``self``/``cls`` and the single leading positional data arg. Params
    are "loose" only if they carry a default (i.e. are optional kwargs the
    convention wants keyword-only).
    """
    args = node.args
    pos = list(args.posonlyargs) + list(args.args)
    if pos and pos[0].arg in ("self", "cls"):
        pos = pos[1:]
    # Defaults align to the tail of `pos`.
    n_defaults = len(args.defaults)
    if n_defaults == 0:
        return 0
    defaulted = pos[len(pos) - n_defaults :]
    return len(defaulted)


def has_star_marker(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if the signature has a ``*`` / ``*args`` before its keyword-only zone."""
    return node.args.vararg is not None or bool(node.args.kwonlyargs)


def check_file(
    path: Path, exempt: dict[tuple[str, str], str]
) -> list[tuple[int, str, int]]:
    rel = rel_posix(path)
    try:
        tree = ast.parse(path.read_text(), filename=rel)
    except SyntaxError as e:  # pragma: no cover - surfaced to caller
        print(f"{rel}: SyntaxError: {e}", file=sys.stderr)
        return []
    violations: list[tuple[int, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") and node.name != "__init__":
            continue
        if _is_exempt(exempt, rel, node.name):
            continue
        if has_star_marker(node):
            continue
        n_loose = loose_kwargs(node)
        if n_loose >= THRESHOLD:
            violations.append((node.lineno, node.name, n_loose))
    return violations


def main(argv: list[str]) -> int:
    roots = argv[1:] or DEFAULT_ROOTS
    exempt = load_exemptions()
    total = 0
    for path in iter_py_files(roots, exclude_parts=EXCLUDE_PARTS):
        for lineno, name, n in check_file(path, exempt):
            print(
                f"{rel_posix(path)}:{lineno}: {name}() has {n} loose kwargs and no "
                f"keyword-only `*` marker (convention: `*` required for 3+ kwargs)"
            )
            total += 1
    if total:
        print(f"\n{total} keyword-only-marker violation(s).", file=sys.stderr)
        return 1
    print("check_kwonly: no violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
