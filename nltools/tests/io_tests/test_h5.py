"""Tests for nltools.io.h5 — HDF5 serialization utilities."""

import os
from pathlib import Path

import numpy as np
import pytest

from nltools.io import is_h5_path, load_brain_data_h5, to_h5


def _tables_available():
    try:
        import tables  # noqa: F401

        return True
    except ImportError:
        return False


class TestIsH5Path:
    """Tests for is_h5_path."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("data.h5", True),
            ("data.hdf5", True),
            ("path/to/file.h5", True),
            ("path/to/file.hdf5", True),
            ("data.csv", False),
            ("data.nii.gz", False),
            ("data.npy", False),
            ("h5_in_name.csv", False),
        ],
    )
    def test_string_paths(self, path, expected):
        assert is_h5_path(path) == expected

    def test_pathlib_path(self):
        assert is_h5_path(Path("results.hdf5")) is True
        assert is_h5_path(Path("results.csv")) is False


_needs_tables = pytest.mark.skipif(
    not _tables_available(), reason="HDF5 support requires PyTables"
)


@_needs_tables
class TestToH5BrainData:
    """Tests for to_h5 with brain_data type (round-trip via fixtures)."""

    def test_invalid_obj_type_raises(self, sim_brain_data, tmp_path):
        with pytest.raises(TypeError, match="obj_type"):
            to_h5(sim_brain_data, str(tmp_path / "bad.h5"), obj_type="invalid")

    def test_round_trip(self, sim_brain_data, tmp_path):
        """Write brain data to h5 and load it back."""
        path = str(tmp_path / "brain.h5")
        to_h5(sim_brain_data, path, obj_type="brain_data")
        assert os.path.exists(path)

        result = load_brain_data_h5(path)
        assert np.allclose(result["data"], sim_brain_data.data)
        assert "load_mask" in result

    def test_round_trip_preserves_mask(self, sim_brain_data, tmp_path):
        """Mask affine and data survive the round-trip."""
        path = str(tmp_path / "brain.h5")
        to_h5(sim_brain_data, path, obj_type="brain_data")

        result = load_brain_data_h5(path)
        assert result["load_mask"] is True
        assert np.allclose(result["mask"].affine, sim_brain_data.mask.affine)
        assert np.allclose(result["mask"].get_fdata(), sim_brain_data.mask.get_fdata())


@_needs_tables
class TestToH5Adjacency:
    """Tests for to_h5 with adjacency type."""

    def test_round_trip(self, sim_adjacency_single, tmp_path):
        """Write adjacency to h5 and verify file exists."""
        path = str(tmp_path / "adj.h5")
        to_h5(sim_adjacency_single, path, obj_type="adjacency")
        assert os.path.exists(path)


@_needs_tables
class TestLoadLegacyH5:
    """Tests for loading legacy (pre-0.4.8) HDF5 files."""

    def test_load_old_brain_h5(self, old_h5_brain, new_h5_brain):
        """Old and new brain h5 files produce equivalent data."""
        with pytest.warns(UserWarning):
            result_old = load_brain_data_h5(old_h5_brain)
        result_new = load_brain_data_h5(new_h5_brain)

        assert np.allclose(result_old["data"], result_new["data"])
        assert result_old["X"].shape == result_new["X"].shape
        assert result_old["Y"].shape == result_new["Y"].shape
