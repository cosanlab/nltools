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
regardless of which layer it lives in. The only exceptions are the explicit,
per-function entries in ``EXEMPT`` below, each with its rationale inline.

Usage:  python scripts/check_kwonly.py [PATH ...]
Exit status 1 if any violations are found (so it can gate CI).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import iter_py_files, rel_posix  # noqa: E402

# Minimum number of loose (defaulted, non-keyword-only) params to require a `*`.
THRESHOLD = 3

# The whole package: the convention is uniform (no per-layer carve-outs).
DEFAULT_ROOTS = ["nltools"]

EXCLUDE_PARTS = {"tests"}

# Documented exemptions: (path, function name) -> rationale. These functions in
# nltools/algorithms/backends.py deliberately mirror numpy signatures so they
# are drop-in substitutes inside backend-generic code (numpy accepts
# ``np.zeros_like(a, dtype, order)`` positionally); forcing a ``*`` here would
# break that substitutability for no safety gain. Do NOT add entries to escape
# the convention for ordinary nltools functions — this list exists only for
# signatures whose shape is dictated by an external API.
EXEMPT: dict[tuple[str, str], str] = {
    ("nltools/algorithms/backends.py", "zeros_like"): "mirrors np.zeros_like",
    ("nltools/algorithms/backends.py", "ones_like"): "mirrors np.ones_like",
    ("nltools/algorithms/backends.py", "full_like"): "mirrors np.full_like",
    ("nltools/algorithms/backends.py", "assert_array_almost_equal"): (
        "mirrors np.testing.assert_array_almost_equal"
    ),
}


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
        if (rel, node.name) in EXEMPT:
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
    for path in iter_py_files(roots, exclude_parts=EXCLUDE_PARTS):
        for lineno, name, n in check_file(path):
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
