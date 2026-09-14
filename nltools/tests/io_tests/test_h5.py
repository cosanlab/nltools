"""Tests for nltools.io.h5 — HDF5 serialization utilities."""

import os
from pathlib import Path

import pytest

from nltools.io.h5 import _is_h5_path, _load_brain_data_h5, _to_h5


class TestIsH5Path:
    """Tests for _is_h5_path."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("data.h5", True),
            ("h5_in_name.csv", False),
        ],
    )
    def test_string_paths(self, path, expected):
        assert _is_h5_path(path) == expected

    def test_pathlib_path(self):
        assert _is_h5_path(Path("results.hdf5")) is True
        assert _is_h5_path(Path("results.csv")) is False


class TestToH5BrainData:
    """Tests for _to_h5 with brain_data type (round-trip via fixtures)."""

    def test_invalid_obj_type_raises(self, sim_brain_data, tmp_path):
        with pytest.raises(ValueError, match="obj_type"):
            _to_h5(sim_brain_data, str(tmp_path / "bad.h5"), obj_type="invalid")


class TestCompressionFilters:
    """Only h5py's own filters are accepted; a plugin name is refused."""

    def test_plugin_filter_name_is_rejected(self, sim_brain_data, tmp_path):
        with pytest.raises(ValueError, match="h5_compression must be one of"):
            _to_h5(
                sim_brain_data,
                str(tmp_path / "blosc.h5"),
                obj_type="brain_data",
                h5_compression="blosc",
            )


class TestToH5Adjacency:
    """Tests for _to_h5 with adjacency type."""

    def test_round_trip(self, sim_adjacency_single, tmp_path):
        """Write adjacency to h5 and verify file exists."""
        path = str(tmp_path / "adj.h5")
        _to_h5(sim_adjacency_single, path, obj_type="adjacency")
        assert os.path.exists(path)

    def test_mixed_type_labels_round_trip_as_text(self, tmp_path):
        """Labels the constructor accepts are writable; mixed ones become text."""
        import numpy as np

        from nltools.data import Adjacency

        adj = Adjacency(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            matrix_type="similarity",
            labels=[1, "right"],
        )
        path = str(tmp_path / "mixed_labels.h5")
        _to_h5(adj, path, obj_type="adjacency")
        assert list(Adjacency(path).labels) == ["1", "right"]


class TestFitRecordDesignSpaces:
    """Banded feature-space names are user text, not HDF5 paths."""

    def test_names_with_slashes_round_trip(self, tmp_path):
        from types import SimpleNamespace

        import numpy as np

        from nltools.io.h5 import _read_fit_record, _write_fit_record

        h5py = pytest.importorskip("h5py")
        design = {
            "a": np.array([[1.0], [2.0]]),
            "a/b": np.array([[3.0], [4.0]]),
            "/abs": np.array([[5.0], [6.0]]),
        }
        fit = SimpleNamespace(
            kind="ridge",
            betas=None,
            predicted=None,
            residual=None,
            r2=None,
            alpha=None,
            design=design,
        )

        path = str(tmp_path / "banded.h5")
        with h5py.File(path, "w") as f:
            _write_fit_record(f, fit, "gzip")
        with h5py.File(path, "r") as f:
            # Nothing escaped the model group into the file root.
            assert set(f.keys()) == {"model"}
            record = _read_fit_record(f)

        assert list(record["design"]) == ["a", "a/b", "/abs"]
        for name, matrix in design.items():
            np.testing.assert_allclose(record["design"][name], matrix)


class TestLegacyLayoutRejected:
    """Files written by nltools 0.5.1 and earlier are refused with an export hint."""

    def test_legacy_brain_data_layout_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "legacy_braindata.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=[[1.0, 2.0]])
            f.create_dataset("X_columns", data=[b"regressor"])

        with pytest.raises(ValueError, match="nltools 0.5.1"):
            _load_brain_data_h5(str(path))

    def test_legacy_adjacency_layout_raises(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from nltools.data import Adjacency

        path = tmp_path / "legacy_adjacency.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=[1.0, 2.0, 3.0])
            f.create_dataset("Y_columns", data=[b"condition"])

        with pytest.raises(ValueError, match="nltools 0.5.1"):
            Adjacency(str(path))


class TestMaskFileName:
    """A stored mask filename is always reduced to its basename on load."""

    def test_windows_mask_path_is_reduced_to_its_basename(
        self, sim_brain_data, tmp_path
    ):
        """A file written on Windows carries backslash separators."""
        h5py = pytest.importorskip("h5py")
        path = tmp_path / "windows_mask_name.h5"
        _to_h5(sim_brain_data, str(path), obj_type="brain_data")
        with h5py.File(path, "a") as f:
            del f["mask_file_name"]
            f.create_dataset(
                "mask_file_name", data="C:\\Users\\someone\\masks\\2mm-mask.nii.gz"
            )

        result = _load_brain_data_h5(str(path))

        assert result["mask"].get_filename() == "2mm-mask.nii.gz"
