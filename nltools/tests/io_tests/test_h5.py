"""Tests for nltools.io.h5 — HDF5 serialization utilities."""

import os
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl
import pytest

from nltools.io import is_h5_path, load_brain_data_h5, to_h5

LEGACY_FIXTURES = Path(__file__).parent / "legacy_fixtures"
LEGACY_BRAINDATA = LEGACY_FIXTURES / "legacy_braindata.h5"
LEGACY_ADJACENCY = LEGACY_FIXTURES / "legacy_adjacency.h5"

requires_legacy_braindata = pytest.mark.skipif(
    not LEGACY_BRAINDATA.exists(),
    reason="legacy braindata fixture not present (gitignored, copy from scripts/)",
)
requires_legacy_adjacency = pytest.mark.skipif(
    not LEGACY_ADJACENCY.exists(),
    reason="legacy adjacency fixture not present (gitignored, copy from scripts/)",
)


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
            # F158: substring matches must not misclassify non-h5 paths.
            ("results.h5.summary.csv", False),
            (".h5cache/data.nii", False),
            ("archive.hdf5.tar.gz", False),
        ],
    )
    def test_string_paths(self, path, expected):
        assert is_h5_path(path) == expected

    def test_pathlib_path(self):
        assert is_h5_path(Path("results.hdf5")) is True
        assert is_h5_path(Path("results.csv")) is False


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


class TestToH5Adjacency:
    """Tests for to_h5 with adjacency type."""

    def test_round_trip(self, sim_adjacency_single, tmp_path):
        """Write adjacency to h5 and verify file exists."""
        path = str(tmp_path / "adj.h5")
        to_h5(sim_adjacency_single, path, obj_type="adjacency")
        assert os.path.exists(path)


@requires_legacy_braindata
class TestLegacyBrainDataH5:
    """Read deepdish/PyTables-format BrainData files written by nltools <= 0.5.1."""

    def test_loads_via_load_brain_data_h5(self):
        result = load_brain_data_h5(str(LEGACY_BRAINDATA))
        assert result["data"].shape == (25, 51029)
        assert result["data"].dtype == np.float64
        assert isinstance(result["X"], pl.DataFrame)
        assert isinstance(result["Y"], pl.DataFrame)
        assert result["X"].is_empty()
        assert result["Y"].is_empty()

    def test_mask_reconstructed_without_mask_file_name(self):
        result = load_brain_data_h5(str(LEGACY_BRAINDATA))
        assert result["load_mask"] is True
        assert isinstance(result["mask"], nib.Nifti1Image)
        assert result["mask"].shape == (61, 73, 61)

    def test_braindata_constructor(self):
        from nltools.data import BrainData

        bd = BrainData(str(LEGACY_BRAINDATA))
        assert bd.data.shape == (25, 51029)
        assert bd.X.is_empty()
        assert bd.Y.is_empty()
        assert bd.mask.shape == (61, 73, 61)


@requires_legacy_adjacency
class TestLegacyAdjacencyH5:
    """Read deepdish/PyTables-format Adjacency files written by nltools <= 0.5.1."""

    def test_adjacency_constructor_defaults_matrix_type(self):
        from nltools.data import Adjacency

        with pytest.warns(UserWarning, match="matrix_type"):
            adj = Adjacency(str(LEGACY_ADJACENCY))
        assert adj.matrix_type == "distance"
        assert adj.labels == []
        assert adj.Y.is_empty()
        assert adj.is_single_matrix is True
        assert adj.issymmetric is True
        # 300-element long-form vector (C(25, 2) = 300) implies 25 nodes
        assert adj.squareform().shape == (25, 25)

    def test_adjacency_constructor_honors_matrix_type_kwarg(self):
        from nltools.data import Adjacency

        # When matrix_type is supplied, no warning should fire
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            adj = Adjacency(str(LEGACY_ADJACENCY), matrix_type="distance")
        assert adj.matrix_type == "distance"
        assert adj.squareform().shape == (25, 25)
