"""Tests for Adjacency I/O: write, read, legacy H5, graph conversion."""

import os
from pathlib import Path

import numpy as np
import networkx as nx
import pytest

from nltools.data import Adjacency
from nltools.tests.conftest import _tables_available


class TestAdjacencyIO:
    def test_write_multiple(self, sim_adjacency_multiple, tmpdir):
        """Test writing and loading multiple adjacency matrices (CSV and HDF5)."""
        sim_adjacency_multiple.write(
            os.path.join(str(tmpdir.join("Test.csv"))), method="long"
        )
        dat_multiple2 = Adjacency(
            os.path.join(str(tmpdir.join("Test.csv"))), matrix_type="distance_flat"
        )
        assert np.all(np.isclose(sim_adjacency_multiple.data, dat_multiple2.data))

        # Test i/o for hdf5 (deprecated - requires PyTables)
        pytest.importorskip(
            "tables", reason="HDF5 support deprecated, requires PyTables"
        )
        sim_adjacency_multiple.write(os.path.join(str(tmpdir.join("test_write.h5"))))
        b = Adjacency(os.path.join(tmpdir.join("test_write.h5")))
        for k in ["Y", "matrix_type", "is_single_matrix", "issymmetric", "data"]:
            if k == "data":
                assert np.allclose(b.__dict__[k], sim_adjacency_multiple.__dict__[k])
            elif k == "Y":
                assert all(b.__dict__[k].eq(sim_adjacency_multiple.__dict__[k]).values)
            else:
                assert b.__dict__[k] == sim_adjacency_multiple.__dict__[k]

    def test_read_and_write_directed(self, sim_adjacency_directed, tmpdir):
        """Test reading and writing directed matrices with Path support."""
        sim_adjacency_directed.write(
            os.path.join(str(tmpdir.join("Test.csv"))), method="long"
        )
        dat_directed2 = Adjacency(
            os.path.join(str(tmpdir.join("Test.csv"))), matrix_type="directed_flat"
        )
        assert np.all(np.isclose(sim_adjacency_directed.data, dat_directed2.data))

        # Load Path
        dat_directed2 = Adjacency(
            Path(tmpdir.join("Test.csv")), matrix_type="directed_flat"
        )
        assert np.all(np.isclose(sim_adjacency_directed.data, dat_directed2.data))

    @pytest.mark.skipif(
        not _tables_available(), reason="HDF5 support deprecated, requires PyTables"
    )
    def test_load_legacy_h5(
        self,
        old_h5_adj_single,
        new_h5_adj_single,
        old_h5_adj_double,
        new_h5_adj_double,
        tmpdir,
    ):
        """Test loading old HDF5 format (backward compatibility)."""
        with pytest.warns(UserWarning):
            b_old = Adjacency(old_h5_adj_single, verbose=True)
        b_new = Adjacency(new_h5_adj_single)
        assert b_old.shape == b_new.shape
        assert np.allclose(b_old.data, b_new.data)
        assert b_old.Y.shape == b_new.Y.shape
        assert b_old.matrix_type == b_new.matrix_type
        assert b_old.is_single_matrix == b_new.is_single_matrix
        assert b_old.issymmetric == b_new.issymmetric
        assert b_old.labels == b_new.labels

        b_old = Adjacency(old_h5_adj_double, legacy_h5=True)
        b_new = Adjacency(new_h5_adj_double)
        assert b_old.shape == b_new.shape
        assert np.allclose(b_old.data, b_new.data)
        assert b_old.Y.shape == b_new.Y.shape
        assert b_old.matrix_type == b_new.matrix_type
        assert b_old.is_single_matrix == b_new.is_single_matrix
        assert b_old.issymmetric == b_new.issymmetric
        assert b_old.labels == b_new.labels

        new_file = Path(tmpdir) / "tmp.h5"
        b_new.write(new_file)
        b_new_written = Adjacency(new_file)
        assert b_new.shape == b_new_written.shape
        assert np.allclose(b_new.data, b_new_written.data)
        new_file.unlink()

    def test_graph_conversion(self, sim_adjacency_single, sim_adjacency_directed):
        """Test conversion to NetworkX graph (directed and undirected)."""
        assert isinstance(sim_adjacency_single.to_graph(), nx.Graph)
        assert isinstance(sim_adjacency_directed.to_graph(), nx.DiGraph)
