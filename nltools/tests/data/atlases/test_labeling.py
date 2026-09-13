"""Tests for coordinate-level atlas labeling."""

import numpy as np
import polars as pl
import pytest

from nltools.data.atlases import label_coords


@pytest.fixture
def known_coords():
    """A few well-known MNI coords for sanity checks.

    All in MNI152 mm space:
    - left M1 hand:    (-42, -22,  56)
    - posterior cing.: (  0, -50,  30)
    - origin:          (  0,   0,   0)  (often unlabeled / outside GM)
    """
    return np.array(
        [
            [-42, -22, 56],
            [0, -50, 30],
            [0, 0, 0],
        ],
        dtype=float,
    )


def test_label_coords_includes_xyz(known_coords):
    df = label_coords(known_coords, atlas="aal")
    assert {"x", "y", "z"}.issubset(df.columns)
    np.testing.assert_array_equal(df.select(["x", "y", "z"]).to_numpy(), known_coords)


def test_label_coords_multiple_atlases(known_coords):
    df = label_coords(known_coords, atlas=["aal", "harvard_oxford"])
    assert "aal" in df.columns
    assert "harvard_oxford" in df.columns
    assert df.height == 3


def test_label_coords_deterministic_returns_str(known_coords):
    df = label_coords(known_coords, atlas="aal")
    assert df["aal"].dtype == pl.Utf8
    # (-42, -22, 56) is around left M1/S1 hand area — should land in
    # Pre- or Postcentral_L depending on exact voxel grid
    assert df["aal"][0] in {"Precentral_L", "Postcentral_L"}


def test_label_coords_probabilistic_format(known_coords):
    df = label_coords(known_coords, atlas="harvard_oxford", prob_threshold=5.0)
    val = df["harvard_oxford"][0]
    assert isinstance(val, str)
    # Probabilistic atlases produce "<pct>% <name>" entries (or "no_label")
    assert "%" in val or val == "no_label"


def test_label_coords_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        label_coords([[-42, -22]], atlas="aal")  # only 2 dims


def test_label_coords_unknown_atlas_raises():
    with pytest.raises(ValueError, match="unknown atlas"):
        label_coords([[0, 0, 0]], atlas="not_real")


def test_label_coords_probabilistic_threshold_filters():
    """Higher prob_threshold should yield fewer or equal regions reported."""
    coord = np.array([[-42, -22, 56]])
    df_low = label_coords(coord, atlas="harvard_oxford", prob_threshold=1.0)
    df_high = label_coords(coord, atlas="harvard_oxford", prob_threshold=50.0)
    # Number of "% " separators is a proxy for number of regions reported
    n_low = df_low["harvard_oxford"][0].count("%")
    n_high = df_high["harvard_oxford"][0].count("%")
    assert n_high <= n_low
