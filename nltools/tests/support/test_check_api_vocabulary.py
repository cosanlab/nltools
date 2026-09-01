"""Tests for the manifest-driven API vocabulary checker (scripts/check_api_vocabulary.py).

The checker walks public `def` signatures with `ast` and validates them against the
machine-readable rules in `docs/_data/api-vocabulary.yml` (the same file that renders
the docs vocabulary tables): banned alias kwargs and per-kwarg semantic contracts
(reserved defaults, required defaults, keyword-onlyness). It is wired into
`poe lint-api` alongside semgrep and check_kwonly.

The script isn't a package member, so it's loaded by file path via importlib.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "check_api_vocabulary.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_api_vocabulary", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations via sys.modules[cls.__module__].
    sys.modules["check_api_vocabulary"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def enforcement(checker):
    return checker.load_enforcement()


def _check(checker, enforcement, src, path="nltools/data/example.py"):
    return checker.check_source(src, path, enforcement)


class TestBannedKwargs:
    def test_banned_alias_flagged_with_canonical(self, checker, enforcement):
        v = _check(checker, enforcement, 'def f(x, algorithm="svd"):\n    pass\n')
        assert len(v) == 1
        assert v[0].kwarg == "algorithm"
        assert "method" in v[0].message

    def test_banned_alias_in_keyword_only_zone_flagged(self, checker, enforcement):
        v = _check(checker, enforcement, 'def f(x, *, perm_type="ols"):\n    pass\n')
        assert [w.kwarg for w in v] == ["perm_type"]

    def test_private_functions_are_exempt(self, checker, enforcement):
        v = _check(checker, enforcement, 'def _f(x, algorithm="svd"):\n    pass\n')
        assert v == []

    def test_methods_inside_classes_are_checked(self, checker, enforcement):
        src = "class A:\n    def f(self, kind=None):\n        pass\n"
        v = _check(checker, enforcement, src)
        assert len(v) == 1
        assert v[0].qualname == "A.f"

    def test_excluded_paths_are_skipped(self, checker, enforcement):
        src = 'def f(x, parallel="cpu"):\n    pass\n'
        assert (
            _check(checker, enforcement, src, "nltools/algorithms/ridge/core.py") == []
        )
        assert len(_check(checker, enforcement, src, "nltools/data/foo.py")) == 1

    def test_backend_kwarg_allowed_only_in_backends_module(self, checker, enforcement):
        src = 'def resolve(x, backend="numpy"):\n    pass\n'
        assert _check(checker, enforcement, src, "nltools/algorithms/backends.py") == []
        assert len(_check(checker, enforcement, src, "nltools/data/foo.py")) == 1


class TestKwargContracts:
    def test_metric_must_not_default_to_central_tendency(self, checker, enforcement):
        v = _check(checker, enforcement, 'def f(x, metric="mean"):\n    pass\n')
        assert len(v) == 1
        assert "summary" in v[0].message

    def test_metric_similarity_default_is_fine(self, checker, enforcement):
        v = _check(checker, enforcement, 'def f(x, metric="correlation"):\n    pass\n')
        assert v == []

    def test_summary_default_must_be_central_tendency(self, checker, enforcement):
        v = _check(checker, enforcement, 'def f(x, *, summary="within"):\n    pass\n')
        assert len(v) == 1
        v = _check(checker, enforcement, 'def f(x, *, summary="median"):\n    pass\n')
        assert v == []

    def test_progress_bar_must_default_false_and_be_keyword_only(
        self, checker, enforcement
    ):
        v = _check(checker, enforcement, "def f(x, *, progress_bar=True):\n    pass\n")
        assert len(v) == 1
        v = _check(checker, enforcement, "def f(x, progress_bar=False):\n    pass\n")
        assert len(v) == 1
        v = _check(checker, enforcement, "def f(x, *, progress_bar=False):\n    pass\n")
        assert v == []

    def test_n_jobs_defaults_to_all_cores(self, checker, enforcement):
        v = _check(checker, enforcement, "def f(x, *, n_jobs=1):\n    pass\n")
        assert len(v) == 1
        v = _check(checker, enforcement, "def f(x, *, n_jobs=-1):\n    pass\n")
        assert v == []

    def test_kwarg_without_default_is_not_contract_checked(self, checker, enforcement):
        v = _check(checker, enforcement, "def f(x, *, summary):\n    pass\n")
        assert v == []

    def test_contract_exclude_paths_are_honored(self, checker, enforcement):
        src = (
            "class Backend:\n    def asarray(self, x, *, device=None):\n        pass\n"
        )
        assert _check(checker, enforcement, src, "nltools/algorithms/backends.py") == []
        assert len(_check(checker, enforcement, src, "nltools/data/foo.py")) == 1


class TestExemptions:
    def test_yaml_exemption_suppresses_violation(self, checker, enforcement):
        import copy

        e = copy.deepcopy(enforcement)
        e["suppressions"].add(("nltools/data/foo.py", "A.f", "kind"))
        src = "class A:\n    def f(self, kind=None):\n        pass\n"
        assert checker.check_source(src, "nltools/data/foo.py", e) == []

    def test_exemption_is_path_qualified(self, checker, enforcement):
        """A suppression binds to its module, not to every same-named def (C-13).

        A bare (function, kwarg) key would exempt ANY future module-level
        `predict` anywhere in the package.
        """
        import copy

        e = copy.deepcopy(enforcement)
        e["suppressions"].add(("nltools/data/foo.py", "A.f", "kind"))
        src = "class A:\n    def f(self, kind=None):\n        pass\n"
        v = checker.check_source(src, "nltools/data/elsewhere.py", e)
        assert [w.kwarg for w in v] == ["kind"]

    def test_exemption_path_is_a_prefix(self, checker, enforcement):
        """Paths match by prefix, like enforcement exclude_paths."""
        import copy

        e = copy.deepcopy(enforcement)
        e["suppressions"].add(("nltools/data/", "A.f", "kind"))
        src = "class A:\n    def f(self, kind=None):\n        pass\n"
        assert checker.check_source(src, "nltools/data/foo.py", e) == []
        assert len(checker.check_source(src, "nltools/models/foo.py", e)) == 1

    def test_manifest_entries_carry_paths(self, enforcement):
        """Every suppression loaded from the YAML is a (path, qualname, kwarg) triple."""
        assert enforcement["suppressions"], "manifest should define suppressions"
        for entry in enforcement["suppressions"]:
            assert len(entry) == 3
            path = entry[0]
            assert path.startswith("nltools/"), entry


class TestRealTree:
    def test_nltools_package_is_vocabulary_clean(self, checker, enforcement):
        violations = checker.check_tree(["nltools"], enforcement)
        assert violations == [], "\n".join(
            f"{v.path}:{v.lineno}: {v.qualname}({v.kwarg}=): {v.message}"
            for v in violations
        )
