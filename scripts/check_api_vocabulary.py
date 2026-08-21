#!/usr/bin/env python3
"""Enforce the canonical-kwarg vocabulary manifest against public signatures.

`docs/_data/api-vocabulary.yml` is the single source of truth for the v0.6.0
canonical-kwarg vocabulary. Its `enforcement:` section is machine-checkable:
this script walks every public (non-underscore) ``def`` in ``nltools/`` with
``ast`` and validates each signature against two rule families:

- **banned_kwargs** — alias parameter names that may not appear at all
  (``algorithm``/``scheme``/``kind`` → ``method``, ``parallel`` →
  ``n_jobs``/``device``, ``show_progress`` → ``progress_bar``, ...).
- **kwarg_contracts** — semantic constraints on canonical kwargs when they
  carry a literal default: ``metric=`` may never default to ``'mean'``/
  ``'median'`` (a central tendency is ``summary=``), ``summary=`` must default
  to one, ``progress_bar`` must be keyword-only ``= False``, ``n_jobs = -1``,
  and so on. These catch the drift semgrep patterns cannot express — a
  correctly-named kwarg holding the wrong concept.

The scope model mirrors ``.semgrep/rules.yml``: tests and the
ridge/alignment/pipelines subsystems are path-excluded because their facades
translate legacy names at the boundary (CLAUDE.md "Facade translation" rule).
Suppressions come from the manifest itself — the documented ``exceptions:``
entries and the ``enforcement.exemptions:`` list — never from inline comments,
so every carve-out is visible in one file.

Wired into ``poe lint-api`` alongside semgrep (result-key and ``**kwargs``
rules) and ``check_kwonly.py`` (the structural ``*``-marker check).

Usage:  python scripts/check_api_vocabulary.py [PATH ...]
Exit status 1 if any violations are found (so it can gate CI).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB_YML = PROJECT_ROOT / "docs" / "_data" / "api-vocabulary.yml"

DEFAULT_ROOTS = ["nltools"]


@dataclass(frozen=True)
class Violation:
    path: str
    lineno: int
    qualname: str
    kwarg: str
    rule: str
    message: str


def load_enforcement(vocab_yml: Path = VOCAB_YML) -> dict:
    """Load the enforcement rules + suppression set from the vocabulary manifest.

    Returns the `enforcement:` mapping with one derived key added:
    `suppressions`, a set of `(qualname, kwarg)` pairs merged from the
    documented `exceptions:` entries and `enforcement.exemptions:`.
    """
    with vocab_yml.open() as f:
        vocab = yaml.safe_load(f)
    enforcement = vocab["enforcement"]
    suppressions: set[tuple[str, str]] = set()
    for entry in list(vocab.get("exceptions", [])) + list(
        enforcement.get("exemptions", [])
    ):
        suppressions.add((entry["function"], entry["kwarg"]))
    enforcement["suppressions"] = suppressions
    return enforcement


def _is_excluded(path: str, prefixes: list[str]) -> bool:
    posix = Path(path).as_posix()
    return any(posix.startswith(prefix) for prefix in prefixes)


def _literal_default(node: ast.expr):
    """Evaluate a default-value node to a Python literal, or raise ValueError."""
    return ast.literal_eval(node)


def _iter_params(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Yield (name, default_node | None, is_keyword_only) for every named parameter."""
    args = node.args
    pos = list(args.posonlyargs) + list(args.args)
    # Defaults align to the tail of the positional parameters.
    pos_defaults: list[ast.expr | None] = [None] * (len(pos) - len(args.defaults))
    pos_defaults += list(args.defaults)
    for arg, default in zip(pos, pos_defaults):
        if arg.arg in ("self", "cls"):
            continue
        yield arg.arg, default, False
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        yield arg.arg, default, True


def _check_def(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
    path: str,
    enforcement: dict,
) -> list[Violation]:
    violations: list[Violation] = []
    banned: dict = enforcement["banned_kwargs"]
    contracts: dict = enforcement["kwarg_contracts"]
    suppressions: set = enforcement["suppressions"]

    for name, default, is_kwonly in _iter_params(node):
        if (qualname, name) in suppressions:
            continue

        rule = banned.get(name)
        if rule is not None and not _is_excluded(path, rule.get("exclude_paths", [])):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    qualname,
                    name,
                    "banned-kwarg",
                    f"`{name}=` is a banned alias — the canonical name is "
                    f"{rule['canonical']}",
                )
            )
            continue

        contract = contracts.get(name)
        if contract is None or _is_excluded(path, contract.get("exclude_paths", [])):
            continue
        if contract.get("keyword_only") and not is_kwonly:
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    qualname,
                    name,
                    "contract-keyword-only",
                    f"`{name}=` must be keyword-only — {contract['hint']}",
                )
            )
        if default is None:
            continue  # required kwargs carry no default to check
        try:
            value = _literal_default(default)
        except ValueError:
            continue  # dynamic default (call, attribute, ...) — not checkable
        if value in contract.get("banned_defaults", []):
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    qualname,
                    name,
                    "contract-banned-default",
                    f"`{name}={value!r}` — {contract['hint']}",
                )
            )
        allowed = contract.get("allowed_defaults")
        if allowed is not None and value not in allowed:
            violations.append(
                Violation(
                    path,
                    node.lineno,
                    qualname,
                    name,
                    "contract-default",
                    f"`{name}={value!r}` is not an allowed default "
                    f"({', '.join(repr(a) for a in allowed)}) — {contract['hint']}",
                )
            )
    return violations


class _Walker(ast.NodeVisitor):
    """Collect violations from public defs, tracking class-qualified names."""

    def __init__(self, path: str, enforcement: dict):
        self.path = path
        self.enforcement = enforcement
        self.stack: list[str] = []
        self.violations: list[Violation] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        public = not node.name.startswith("_") or node.name == "__init__"
        if public:
            qualname = ".".join([*self.stack, node.name])
            self.violations.extend(
                _check_def(node, qualname, self.path, self.enforcement)
            )
        # Still descend: nested defs inside private functions stay in scope.
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def


def check_source(src: str, path: str, enforcement: dict) -> list[Violation]:
    """Check one module's source text; `path` decides path-based exclusions."""
    if _is_excluded(path, enforcement.get("exclude_paths", [])):
        return []
    walker = _Walker(path, enforcement)
    walker.visit(ast.parse(src, filename=path))
    return walker.violations


def check_tree(roots: list[str], enforcement: dict) -> list[Violation]:
    """Check every .py file under `roots` (paths relative to the project root)."""
    violations: list[Violation] = []
    for root in roots:
        base = PROJECT_ROOT / root
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for f in files:
            rel = f.relative_to(PROJECT_ROOT).as_posix()
            violations.extend(check_source(f.read_text(), rel, enforcement))
    return violations


def main(argv: list[str]) -> int:
    enforcement = load_enforcement()
    violations = check_tree(argv[1:] or DEFAULT_ROOTS, enforcement)
    for v in violations:
        print(f"{v.path}:{v.lineno}: {v.qualname}(): [{v.rule}] {v.message}")
    if violations:
        print(
            f"\n{len(violations)} vocabulary violation(s). Canonical names live in "
            "docs/_data/api-vocabulary.yml (and CLAUDE.md's API table); documented "
            "carve-outs go in that file's `exceptions:`/`enforcement.exemptions:`.",
            file=sys.stderr,
        )
        return 1
    print("check_api_vocabulary: no violations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
