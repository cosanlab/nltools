"""Fast API-convention tests for the alignment estimators.

These pin keyword-only signatures (F005) and the progress_bar default (F004)
without running the (slow) alignment algorithms.
"""

import numpy as np
import pytest

from nltools.algorithms.alignment import SRM, DetSRM, HyperAlignment, LocalAlignment


# ========== F005: keyword-only `*` marker on public estimator methods ==========


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_fit_rejects_positional_options(cls):
    """fit() must reject parallel passed positionally (binding fails pre-compute)."""
    X = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        cls().fit(X, None, "cpu")


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_transform_rejects_positional_options(cls):
    """transform() must reject parallel passed positionally."""
    X = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        cls().transform(X, None, "cpu")


def test_hyperalignment_fit_rejects_positional_options():
    """HyperAlignment.fit() must reject parallel passed positionally."""
    data = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        HyperAlignment().fit(data, "cpu")


def test_hyperalignment_transform_rejects_positional_options():
    """HyperAlignment.transform() must reject parallel passed positionally."""
    data = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        HyperAlignment().transform(data, "cpu")


# ========== F004: LocalAlignment progress_bar defaults to off ==========


def test_localalignment_progress_bar_defaults_off():
    """LocalAlignment must expose progress_bar defaulting to False."""
    la = LocalAlignment()
    assert la.progress_bar is False


# ========== gdzq: alignment vocabulary renames ==========
# SRM/DetSRM/HyperAlignment/LocalAlignment are exported directly with no
# facade, so `rand_seed`/`features`/`max_memory_gb` are public vocabulary.
# Canonical names: random_state, n_features, memory_budget_gb; n_iter stays.


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


def test_localalignment_accepts_memory_budget_gb():
    """LocalAlignment accepts memory_budget_gb (the canonical name)."""
    la = LocalAlignment(memory_budget_gb=2.0)
    assert la.memory_budget_gb == 2.0


def test_localalignment_rejects_legacy_max_memory_gb_kwarg():
    """`max_memory_gb=` is a removed legacy alias; the canonical name is memory_budget_gb."""
    with pytest.raises(TypeError):
        LocalAlignment(max_memory_gb=2.0)
