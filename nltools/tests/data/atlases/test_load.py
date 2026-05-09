"""Tests for atlas loading from the HF dataset."""

import nibabel as nb
import polars as pl
import pytest

from nltools.data.atlases import Atlas, load_atlas


@pytest.mark.parametrize("name", ["aal", "harvard_oxford"])
def test_load_atlas_returns_Atlas(name):
    atlas = load_atlas(name)
    assert isinstance(atlas, Atlas)
    assert atlas.name == name


@pytest.mark.parametrize(
    "name,expected_kind",
    [("aal", "deterministic"), ("harvard_oxford", "probabilistic")],
)
def test_load_atlas_kind(name, expected_kind):
    atlas = load_atlas(name)
    assert atlas.kind == expected_kind


def test_load_atlas_image_is_nifti():
    atlas = load_atlas("aal")
    assert isinstance(atlas.image, nb.Nifti1Image)
    # AAL is 3D deterministic
    assert len(atlas.image.shape) == 3


def test_load_atlas_probabilistic_image_is_4d():
    atlas = load_atlas("harvard_oxford")
    assert len(atlas.image.shape) == 4


def test_load_atlas_labels_is_polars_dataframe():
    atlas = load_atlas("aal")
    assert isinstance(atlas.labels, pl.DataFrame)
    assert atlas.labels.columns == ["index", "name"]
    # AAL has 120 regions (per the CSV in the HF dataset)
    assert atlas.labels.height == 120


def test_load_atlas_is_cached():
    """Second call should return the same instance (no re-fetch)."""
    a1 = load_atlas("aal")
    a2 = load_atlas("aal")
    assert a1 is a2


def test_load_atlas_unknown_raises():
    with pytest.raises(ValueError, match="unknown atlas"):
        load_atlas("not_a_real_atlas")


def test_atlas_is_frozen():
    atlas = load_atlas("aal")
    with pytest.raises((AttributeError, TypeError)):
        atlas.name = "other"  # type: ignore[misc]


def test_atlas_has_metadata_citation():
    atlas = load_atlas("aal")
    assert "Tzourio" in atlas.citation
