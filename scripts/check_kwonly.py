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

Scope: the public API surface where the convention binds — the four data-class
facades plus public stats/models/mask/datasets functions. The algorithm-layer
internals (``nltools/algorithms``, ``nltools/pipelines``) are excluded; a handful
of public estimators there (SRM/Ridge/Glm) are covered because ``models/`` is in
scope.

Usage:  python scripts/check_kwonly.py [PATH ...]
Exit status 1 if any violations are found (so it can gate CI).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Minimum number of loose (defaulted, non-keyword-only) params to require a `*`.
THRESHOLD = 3

# Public API roots where the convention binds. Algorithm/pipeline internals are
# intentionally excluded (facade-translation rule); models/ is included because
# Ridge/Glm are public estimators the audit flagged (F103, F106).
DEFAULT_ROOTS = [
    "nltools/data",
    "nltools/stats",
    "nltools/models",
    "nltools/mask.py",
    "nltools/datasets.py",
]

EXCLUDE_PARTS = {"tests"}


def iter_py_files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        p = Path(root)
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(
                f
                for f in p.rglob("*.py")
                if not (EXCLUDE_PARTS & set(f.parts))
            )
    return files


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


def check_file(path: Path) -> list[tuple[int, str, int]]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:  # pragma: no cover - surfaced to caller
        print(f"{path}: SyntaxError: {e}", file=sys.stderr)
        return []
    violations: list[tuple[int, str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") and node.name != "__init__":
            continue
        if has_star_marker(node):
            continue
        n_loose = loose_kwargs(node)
        if n_loose >= THRESHOLD:
            violations.append((node.lineno, node.name, n_loose))
    return violations


def main(argv: list[str]) -> int:
    roots = argv[1:] or DEFAULT_ROOTS
    total = 0
    for path in sorted(iter_py_files(roots)):
        for lineno, name, n in check_file(path):
            print(
                f"{path}:{lineno}: {name}() has {n} loose kwargs and no "
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
