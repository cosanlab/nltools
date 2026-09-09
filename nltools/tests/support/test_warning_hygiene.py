"""Warning hygiene across the library.

Three rules, all enforced by scanning the package source (like the budget-math
scan in ``core/test_backends.py``):

1. No module-level ``warnings.filterwarnings`` / ``warnings.simplefilter``:
   an import-time filter silences a dependency for every program that
   imports nltools. Filters belong inside a function, scoped by
   ``warnings.catch_warnings()``.
2. Every ``warnings.warn(...)`` names its category explicitly — an implicit
   ``UserWarning`` cannot be silenced by class and reads as an accident.
3. Every ``warnings.warn(...)`` passes ``stacklevel=find_stack_level()`` so
   the warning is attributed to the user's line, not to a frame inside
   nltools (fixed integers drift as facades gain layers).

Plus behavioral tests for ``find_stack_level`` itself.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pytest

import nltools
from nltools.utils import DesignMatrixWarning, find_stack_level

PACKAGE_ROOT = Path(nltools.__file__).parent


def _library_sources():
    for py in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in py.relative_to(PACKAGE_ROOT).parts:
            continue
        yield py, ast.parse(py.read_text(), filename=str(py))


def _is_warnings_call(node: ast.AST, names: set[str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "warnings"
        and node.func.attr in names
    )


def _module_level_statements(tree: ast.Module):
    """Yield statements that run at import time (not inside a def/class)."""
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield node
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                stack.append(child)


class TestNoImportTimeFilters:
    def test_no_module_level_filterwarnings_or_simplefilter(self):
        offenders = []
        for py, tree in _library_sources():
            for stmt in _module_level_statements(tree):
                for node in ast.walk(stmt):
                    if _is_warnings_call(node, {"filterwarnings", "simplefilter"}):
                        offenders.append(
                            f"{py.relative_to(PACKAGE_ROOT)}:{node.lineno}"
                        )
        assert not offenders, (
            "Import-time warning filters silence a dependency for every program "
            "that imports nltools. Scope filters inside a function with "
            "warnings.catch_warnings() instead. Offenders:\n" + "\n".join(offenders)
        )


class TestEveryWarnIsAttributedAndCategorized:
    def _warn_calls(self):
        for py, tree in _library_sources():
            for node in ast.walk(tree):
                if _is_warnings_call(node, {"warn"}):
                    yield f"{py.relative_to(PACKAGE_ROOT)}:{node.lineno}", node

    def test_every_warn_names_a_category(self):
        offenders = []
        for where, call in self._warn_calls():
            has_positional_category = len(call.args) >= 2
            has_keyword_category = any(k.arg == "category" for k in call.keywords)
            if not (has_positional_category or has_keyword_category):
                offenders.append(where)
        assert not offenders, (
            "warnings.warn() without an explicit category emits a bare "
            "UserWarning. Name the category. Offenders:\n" + "\n".join(offenders)
        )

    def test_every_warn_uses_find_stack_level(self):
        offenders = []
        for where, call in self._warn_calls():
            stacklevel = next(
                (k.value for k in call.keywords if k.arg == "stacklevel"), None
            )
            ok = (
                isinstance(stacklevel, ast.Call)
                and isinstance(stacklevel.func, ast.Name)
                and stacklevel.func.id == "find_stack_level"
            )
            if not ok:
                offenders.append(where)
        assert not offenders, (
            "warnings.warn() must pass stacklevel=find_stack_level() so the "
            "warning lands on the caller's line outside nltools. Offenders:\n"
            + "\n".join(offenders)
        )


class TestFindStackLevel:
    def test_test_files_count_as_outside_the_package(self):
        """nltools/tests/ lives under the package dir but is user code here."""
        # Only find_stack_level's own frame is inside the library -> level 1,
        # so a warn() issued from this file is attributed to this file.
        assert find_stack_level() == 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.warn("probe", UserWarning, stacklevel=find_stack_level())
        assert caught[0].filename == __file__

    def test_walks_past_library_frames(self):
        """A warning raised inside the library is attributed to the caller."""
        from nltools.data import DesignMatrix

        dm = DesignMatrix({"a": [1.0, 2.0, 3.0, 4.0]}, TR=1.0).add_poly(order=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm.add_poly(order=0)  # "already has 0th order polynomial" notice
        assert caught, "expected the already-exists notice"
        assert caught[0].filename == __file__
        assert caught[0].category is DesignMatrixWarning

    @pytest.mark.xfail(reason="te1m: facade alignment pending", strict=True)
    def test_skips_contextlib_decorator_frames(self):
        """`@coalesced_gc()` facades put a stdlib contextlib frame between the
        user and nltools; the level must step over it too."""
        import nibabel as nib
        import numpy as np

        from nltools.data import BrainData
        from nltools.data.braindata.modeling import RankDeficientDesignWarning

        mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int8), np.eye(4))
        rng = np.random.default_rng(0)
        bd = BrainData(
            nib.Nifti1Image(rng.standard_normal((3, 3, 3, 6)), np.eye(4)), mask=mask
        )
        design = np.column_stack([np.arange(6.0), 2 * np.arange(6.0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bd.fit(model="glm", X=design)  # BrainData.fit is @coalesced_gc()
        rank = [w for w in caught if w.category is RankDeficientDesignWarning]
        assert rank and rank[0].filename == __file__
