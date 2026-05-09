"""Tests for the atlas registry — names, kinds, metadata."""

import pytest

from nltools.data.atlases.registry import (
    ATLASES,
    AtlasKind,
    AtlasMetadata,
    list_atlases,
)


# All 11 atlases hosted at huggingface.co/datasets/nltools/niftis under atlases/.
EXPECTED_ATLASES = {
    "aal",
    "aicha",
    "desikan_killiany",
    "destrieux",
    "harvard_oxford",
    "juelich",
    "marsatlas",
    "neuromorphometrics",
    "schaefer_200",
    "talairach_ba",
    "talairach_gyrus",
}

# Probabilistic atlases store per-region probability maps along the 4th
# dim of the NIfTI; deterministic atlases are integer-valued 3D volumes.
PROBABILISTIC = {"harvard_oxford", "juelich"}


def test_list_atlases_returns_all_eleven():
    names = list_atlases()
    assert set(names) == EXPECTED_ATLASES
    assert names == sorted(names), "list_atlases must return sorted names"


def test_registry_keys_match_expected():
    assert set(ATLASES.keys()) == EXPECTED_ATLASES


@pytest.mark.parametrize("name", sorted(EXPECTED_ATLASES))
def test_metadata_kind_correct(name):
    meta = ATLASES[name]
    assert isinstance(meta, AtlasMetadata)
    expected_kind: AtlasKind = (
        "probabilistic" if name in PROBABILISTIC else "deterministic"
    )
    assert meta.kind == expected_kind


@pytest.mark.parametrize("name", sorted(EXPECTED_ATLASES))
def test_metadata_has_citation(name):
    meta = ATLASES[name]
    assert meta.citation, f"{name} missing citation"
    assert isinstance(meta.citation, str)


def test_atlas_metadata_is_frozen():
    meta = ATLASES["harvard_oxford"]
    with pytest.raises((AttributeError, TypeError)):
        meta.kind = "deterministic"  # type: ignore[misc]


def test_default_atlases_constant():
    """The trio used as the default for cluster_report."""
    from nltools.data.atlases.registry import DEFAULT_ATLASES

    assert DEFAULT_ATLASES == ("harvard_oxford", "aal", "schaefer_200")
