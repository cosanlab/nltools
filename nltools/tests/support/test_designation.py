"""The user-facing surface is exactly the nine namespaces' `__all__`.

An unprefixed module-level function or class anywhere in `nltools/` is
user-facing by construction (CLAUDE.md), so this is the suite's only
export-structure test: everything else carries a leading underscore.
"""

import ast
import importlib
from pathlib import Path

import nltools

USER_FACING_NAMESPACES = [
    "nltools",
    "nltools.data",
    "nltools.algorithms",
    "nltools.io",
    "nltools.datasets",
    "nltools.mask",
    "nltools.cross_validation",
    "nltools.plotting",
    "nltools.utils",
]

PACKAGE_ROOT = Path(nltools.__file__).parent


def _designated_names():
    """Every name exported by a user-facing namespace's `__all__`."""
    names = set()
    for namespace in USER_FACING_NAMESPACES:
        names.update(importlib.import_module(namespace).__all__)
    return names


def _module_level_definitions():
    """Yield (module path, name) for every module-level `def`/`class` outside tests."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(PACKAGE_ROOT).parts:
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                yield path.relative_to(PACKAGE_ROOT.parent), node.name


def test_every_unprefixed_definition_is_designated():
    """An unprefixed module-level def or class must be on the user-facing surface."""
    designated = _designated_names()
    offenders = [
        f"{module}:{name}"
        for module, name in _module_level_definitions()
        if not name.startswith("_") and name not in designated
    ]
    assert not offenders, (
        "these module-level names are neither underscore-prefixed nor exported by a "
        f"user-facing namespace: {offenders}"
    )


def test_every_designated_name_resolves():
    """No namespace advertises a name it cannot supply."""
    dangling = []
    for namespace in USER_FACING_NAMESPACES:
        module = importlib.import_module(namespace)
        dangling += [
            f"{namespace}.{name}"
            for name in module.__all__
            if not hasattr(module, name)
        ]
    assert not dangling, f"`__all__` advertises names that do not resolve: {dangling}"
