"""F152: every name in nltools.__all__ must be a real, accessible attribute."""

import nltools


def test_all_names_accessible():
    """Accessing each name advertised in __all__ must not raise."""
    for name in nltools.__all__:
        getattr(nltools, name)


def test_datasets_and_cross_validation_importable():
    """The two submodules that were advertised but never imported are reachable."""
    assert nltools.datasets is not None
    assert nltools.cross_validation is not None
