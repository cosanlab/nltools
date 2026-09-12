"""Fast API-convention tests for the alignment estimators.

These pin the canonical keyword vocabulary (gdzq) without running the (slow)
alignment algorithms.
"""

import pytest

from nltools.algorithms.alignment import SRM, DetSRM


# ========== gdzq: alignment vocabulary renames ==========
# SRM/DetSRM are exported directly with no facade, so `rand_seed`/`features`
# are public vocabulary. Canonical names: random_state, n_features; n_iter stays.


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_accepts_canonical_names(cls):
    """SRM/DetSRM accept n_features and random_state (the canonical names)."""
    model = cls(n_iter=3, n_features=5, random_state=7)
    assert model.n_iter == 3
    assert model.n_features == 5
    assert model.random_state == 7


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_rejects_legacy_features_kwarg(cls):
    """`features=` is a removed legacy alias; the canonical name is n_features."""
    with pytest.raises(TypeError):
        cls(features=5)


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_rejects_legacy_rand_seed_kwarg(cls):
    """`rand_seed=` is a removed legacy alias; the canonical name is random_state."""
    with pytest.raises(TypeError):
        cls(rand_seed=7)
