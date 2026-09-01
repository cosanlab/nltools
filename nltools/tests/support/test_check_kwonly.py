"""Tests for the keyword-only `*`-marker checker (scripts/check_kwonly.py).

The checker enforces the v0.6.0 convention that a `*` marker separates 3+
defaulted kwargs. Its carve-outs live in `docs/_data/api-vocabulary.yml`
(`enforcement.kwonly_exemptions:`) — the single suppression home the manifest
contract mandates — not in an inline dict.

The script isn't a package member, so it's loaded by file path via importlib.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "check_kwonly.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_kwonly", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_kwonly"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestManifestExemptions:
    def test_exemptions_load_from_the_vocabulary_manifest(self, checker):
        """The four backends carve-outs come from the YAML, with their reasons."""
        exempt = checker.load_exemptions()
        for fn in (
            "zeros_like",
            "ones_like",
            "full_like",
            "assert_array_almost_equal",
        ):
            key = ("nltools/algorithms/backends.py", fn)
            assert key in exempt
            assert exempt[key]  # a non-empty reason
        # No second suppression home: the inline dict is gone.
        assert not hasattr(checker, "EXEMPT")

    def test_exemption_suppresses_only_its_module(self, checker, tmp_path):
        src = "def zeros_like(a, dtype=None, order='K', subok=True):\n    pass\n"
        f = tmp_path / "mod.py"
        f.write_text(src)

        flagged = checker.check_file(f, exempt={})
        assert [name for _, name, _ in flagged] == ["zeros_like"]

        exempt = {(f.as_posix(), "zeros_like"): "mirrors np.zeros_like"}
        assert checker.check_file(f, exempt=exempt) == []


class TestRealTree:
    def test_nltools_package_is_kwonly_clean(self, checker):
        assert checker.main(["check_kwonly"]) == 0
