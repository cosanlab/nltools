"""F152: every name in nltools.__all__ must be a real, accessible attribute."""

import importlib
import pkgutil
import types

import nltools


def test_all_names_accessible():
    """Accessing each name advertised in __all__ must not raise."""
    for name in nltools.__all__:
        getattr(nltools, name)


def test_datasets_and_cross_validation_importable():
    """The two submodules that were advertised but never imported are reachable."""
    assert nltools.datasets is not None
    assert nltools.cross_validation is not None


def test_every_advertised_submodule_is_attribute_reachable():
    """Every submodule name in `nltools.__all__` must resolve via plain attribute access."""
    submodule_names = {
        name
        for name in nltools.__all__
        if isinstance(getattr(nltools, name, None), types.ModuleType)
    }
    for name in submodule_names:
        submodule = getattr(nltools, name)
        assert submodule.__name__ == f"nltools.{name}"


def test_no_all_advertises_an_underscore_prefixed_name():
    """No `__all__` list anywhere in the package may contain a `_`-prefixed name.

    C1 (q31x mg6z): `nltools/algorithms/inference/__init__.py` advertised
    `_auto_batch_size`, `_compute_pvalue`, and `_generate_sign_flips` in
    `__all__`. AGENTS.md requires user-facing names never carry a `_` prefix.
    """

    def is_private(name):
        return name.startswith("_") and not name.startswith("__")

    # walk_packages never yields the top-level package, so seed with it.
    offenders = [
        f"{nltools.__name__}.{name}" for name in nltools.__all__ if is_private(name)
    ]
    for module_info in pkgutil.walk_packages(
        nltools.__path__, prefix=f"{nltools.__name__}."
    ):
        parts = module_info.name.split(".")
        if "tests" in parts or any(part.startswith("_") for part in parts):
            continue
        module = importlib.import_module(module_info.name)
        for exported_name in getattr(module, "__all__", ()):
            if is_private(exported_name):
                offenders.append(f"{module_info.name}.{exported_name}")
    assert not offenders, (
        f"__all__ in these modules advertises a `_`-prefixed name: {offenders}"
    )
