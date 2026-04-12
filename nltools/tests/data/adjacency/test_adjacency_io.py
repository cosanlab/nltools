"""Tests for Adjacency I/O: write, read, legacy H5, graph conversion."""

import os
from pathlib import Path

import numpy as np
import networkx as nx

from nltools.data import Adjacency


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

        # Test i/o for hdf5 (h5py + polars layout — no PyTables required)
        sim_adjacency_multiple.write(os.path.join(str(tmpdir.join("test_write.h5"))))
        b = Adjacency(os.path.join(tmpdir.join("test_write.h5")))
        assert np.allclose(b.data, sim_adjacency_multiple.data)
        assert b.matrix_type == sim_adjacency_multiple.matrix_type
        assert b.is_single_matrix == sim_adjacency_multiple.is_single_matrix
        assert b.issymmetric == sim_adjacency_multiple.issymmetric
        assert b.Y.equals(sim_adjacency_multiple.Y)
        assert b.labels == sim_adjacency_multiple.labels

    def test_h5_roundtrip_y(self, sim_adjacency_multiple, tmpdir):
        """Y round-trips through the new h5py + polars layout."""
        import polars as pl

        n = len(sim_adjacency_multiple)
        adj = sim_adjacency_multiple.copy()
        adj.Y = pl.DataFrame({"label": np.arange(n, dtype=np.int64)})

        path = os.path.join(str(tmpdir.join("roundtrip_y.h5")))
        adj.write(path)

        loaded = Adjacency(path)
        assert isinstance(loaded.Y, pl.DataFrame)
        assert loaded.Y.shape == (n, 1)
        assert loaded.Y.columns == ["label"]
        assert loaded.Y.equals(adj.Y)

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

    def test_graph_conversion(self, sim_adjacency_single, sim_adjacency_directed):
        """Test conversion to NetworkX graph (directed and undirected)."""
        assert isinstance(sim_adjacency_single.to_graph(), nx.Graph)
        assert isinstance(sim_adjacency_directed.to_graph(), nx.DiGraph)
