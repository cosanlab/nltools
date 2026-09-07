"""Packaging invariants.

These guard against a dependency being *used* unconditionally while only being
*declared* in an optional extra — a failure mode that is invisible in the dev
environment (which installs every extra) and only surfaces for end users.
"""

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Shared persistence imports must be available in a core installation.
PERSISTENCE_MODULE = Path("nltools/io/h5.py")
DISTRIBUTION_FOR_MODULE = {"h5py": "h5py", "hdf5plugin": "hdf5plugin"}


def _core_dependency_names() -> set[str]:
    if not PYPROJECT.exists():  # installed-package test run, no source tree
        pytest.skip("pyproject.toml not available (not a source checkout)")
    with PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    names = set()
    for spec in data["project"]["dependencies"]:
        # "h5py>=3.15" -> "h5py"; "nltools[h5]" -> "nltools"
        name = spec.split(";")[0].strip()
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _unguarded_imports(path: Path) -> set[str]:
    """Top-level module names imported outside any try/except ImportError."""
    tree = ast.parse(path.read_text())
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                if isinstance(child, ast.Import):
                    guarded.update(a.name.split(".")[0] for a in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module:
                    guarded.add(child.module.split(".")[0])
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found - guarded


def test_persistence_imports_are_core_dependencies():
    """Shared HDF5 persistence dependencies belong in the core installation."""
    path = REPO_ROOT / PERSISTENCE_MODULE
    if not path.exists():
        pytest.skip(f"{PERSISTENCE_MODULE} not present")

    core = _core_dependency_names()
    unguarded = _unguarded_imports(path)

    missing = {
        dist
        for module, dist in DISTRIBUTION_FOR_MODULE.items()
        if module in unguarded and dist.lower() not in core
    }
    assert not missing, (
        f"{PERSISTENCE_MODULE} imports {sorted(missing)} unconditionally, but "
        f"they are not in [project].dependencies. Either declare them as core "
        f"dependencies or guard the import with a message pointing at the extra."
    )
