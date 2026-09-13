"""Tests for atlas loading from the HF dataset."""

import polars as pl
import pytest

from nltools.data.atlases import _Atlas, list_atlases, load_atlas

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


def test_list_atlases_returns_all_eleven():
    names = list_atlases()
    assert set(names) == EXPECTED_ATLASES
    assert names == sorted(names), "list_atlases must return sorted names"


@pytest.mark.parametrize("name", ["aal"])
def test_load_atlas_returns_Atlas(name):
    atlas = load_atlas(name)
    assert isinstance(atlas, _Atlas)
    assert atlas.name == name


@pytest.mark.parametrize(
    "name,expected_kind",
    [("harvard_oxford", "probabilistic")],
)
def test_load_atlas_kind(name, expected_kind):
    atlas = load_atlas(name)
    assert atlas.kind == expected_kind


def test_load_atlas_labels_is_polars_dataframe():
    atlas = load_atlas("aal")
    assert isinstance(atlas.labels, pl.DataFrame)
    assert atlas.labels.columns == ["index", "name"]
    # AAL has 120 regions (per the CSV in the HF dataset)
    assert atlas.labels.height == 120


def test_load_atlas_unknown_raises():
    with pytest.raises(ValueError, match="unknown atlas"):
        load_atlas("not_a_real_atlas")


def test_atlas_is_frozen():
    atlas = load_atlas("aal")
    with pytest.raises((AttributeError, TypeError)):
        atlas.name = "other"  # type: ignore[misc]
