"""Tests for Adjacency core operations: init, type inference, properties,
indexing, arithmetic, copy, squareform, append, aggregation, distance."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import pearsonr

from nltools.data import Adjacency


class TestAdjacencyCore:
    def test_type_single(self, sim_adjacency_single):
        """Test symmetric matrix type detection (distance vs similarity)."""
        assert sim_adjacency_single.matrix_type == "distance"
        dat_single2 = Adjacency(1 - sim_adjacency_single.squareform())
        assert dat_single2.matrix_type == "similarity"
        assert sim_adjacency_single.issymmetric

    def test_type_directed(self, sim_adjacency_directed):
        """Test directed matrix initialization."""
        assert not sim_adjacency_directed.issymmetric

    def test_length(self, sim_adjacency_multiple):
        """Test length property for multiple adjacency matrices."""
        assert len(sim_adjacency_multiple) == sim_adjacency_multiple.data.shape[0]
        assert len(sim_adjacency_multiple[0]) == 1

    def test_indexing(self, sim_adjacency_multiple):
        """Test indexing with integers, slices, and tuples."""
        assert len(sim_adjacency_multiple[0]) == 1
        assert len(sim_adjacency_multiple[0:4]) == 4
        assert len(sim_adjacency_multiple[0, 2, 3]) == 3

    def test_arithmetic(self, sim_adjacency_directed):
        """Test arithmetic operations on adjacency matrices."""
        assert (sim_adjacency_directed + 5).data[0] == sim_adjacency_directed.data[
            0
        ] + 5
        assert (sim_adjacency_directed - 0.5).data[0] == sim_adjacency_directed.data[
            0
        ] - 0.5
        assert (sim_adjacency_directed * 5).data[0] == sim_adjacency_directed.data[
            0
        ] * 5
        assert np.all(
            np.isclose(
                (sim_adjacency_directed + sim_adjacency_directed).data,
                (sim_adjacency_directed * 2).data,
            )
        )
        assert np.all(
            np.isclose(
                (sim_adjacency_directed * 2 - sim_adjacency_directed).data,
                sim_adjacency_directed.data,
            )
        )
        np.testing.assert_almost_equal(
            ((2 * sim_adjacency_directed / 2) / sim_adjacency_directed).mean(),
            1,
            decimal=4,
        )

    def test_copy(self, sim_adjacency_multiple):
        """Test copying adjacency matrices."""
        assert np.all(sim_adjacency_multiple.data == sim_adjacency_multiple.copy().data)

    def test_squareform(self, sim_adjacency_multiple):
        """Test vector → matrix → vector conversion preserves data."""
        assert len(sim_adjacency_multiple.squareform()) == len(sim_adjacency_multiple)

    def test_shape_property(self, sim_adjacency_single, sim_adjacency_multiple):
        """Test shape property returns (n_nodes, n_nodes) for single, (n, n_nodes, n_nodes) for stacked."""
        assert sim_adjacency_single.shape == (4, 4)
        assert sim_adjacency_single.n_nodes == 4
        assert sim_adjacency_single.vector_shape == (6,)

        n_matrices = len(sim_adjacency_multiple)
        assert sim_adjacency_multiple.shape == (n_matrices, 4, 4)
        assert sim_adjacency_multiple.n_nodes == 4
        assert sim_adjacency_multiple.vector_shape == (n_matrices, 6)

    def test_append(self, sim_adjacency_single):
        """Test appending adjacency matrices."""
        a = Adjacency()
        a = a.append(sim_adjacency_single)
        assert a.shape == sim_adjacency_single.shape
        a = a.append(a)
        assert a.shape == (2, 4, 4)
        assert a.vector_shape == (2, 6)

    def test_Y_is_polars(self, sim_adjacency_multiple):
        """Adjacency.Y is always a polars DataFrame (possibly empty)."""
        import pandas as pd
        import polars as pl

        # Empty constructor -> empty polars frame
        a = Adjacency()
        assert isinstance(a.Y, pl.DataFrame)
        assert a.Y.is_empty()

        # Fixture without Y -> empty polars frame
        assert isinstance(sim_adjacency_multiple.Y, pl.DataFrame)
        assert sim_adjacency_multiple.Y.is_empty()

        # Pandas Y accepted, stored as polars
        n = len(sim_adjacency_multiple)
        adj = sim_adjacency_multiple.copy()
        adj.Y = pd.DataFrame({"label": np.arange(n)})
        assert isinstance(adj.Y, pl.DataFrame)
        assert adj.Y.shape == (n, 1)

        # Polars Y accepted on the setter
        adj.Y = pl.DataFrame({"label": np.arange(n)})
        assert isinstance(adj.Y, pl.DataFrame)

        # Indexing preserves polars
        sliced = adj[0]
        assert isinstance(sliced.Y, pl.DataFrame)
        assert sliced.Y.shape[0] == 1

        # Slice indexing preserves polars
        sliced2 = adj[0:2]
        assert isinstance(sliced2.Y, pl.DataFrame)
        assert sliced2.Y.shape[0] == 2

        # Append concatenates polars
        combined = adj.append(adj)
        assert isinstance(combined.Y, pl.DataFrame)
        assert combined.Y.shape[0] == 2 * n

    def test_mean(self, sim_adjacency_multiple):
        """Test mean aggregation across adjacency matrices."""
        assert isinstance(sim_adjacency_multiple.mean(axis=0), Adjacency)
        assert len(sim_adjacency_multiple.mean(axis=0)) == 1
        assert len(sim_adjacency_multiple.mean(axis=1)) == len(
            np.mean(sim_adjacency_multiple.data, axis=1)
        )

    def test_std(self, sim_adjacency_multiple):
        """Test standard deviation across adjacency matrices."""
        assert isinstance(sim_adjacency_multiple.std(axis=0), Adjacency)
        assert len(sim_adjacency_multiple.std(axis=0)) == 1
        assert len(sim_adjacency_multiple.std(axis=1)) == len(
            np.std(sim_adjacency_multiple.data, axis=1)
        )

    def test_median(self, sim_adjacency_single, sim_adjacency_multiple):
        """Test median calculation for single and multiple adjacency matrices."""
        single_median = sim_adjacency_single.median()
        assert isinstance(single_median, (float, np.floating))
        assert np.isclose(single_median, np.nanmedian(sim_adjacency_single.data))

        median_axis0 = sim_adjacency_multiple.median(axis=0)
        assert isinstance(median_axis0, Adjacency)
        assert len(median_axis0) == 1
        np.testing.assert_array_almost_equal(
            median_axis0.data.flatten(),
            np.nanmedian(sim_adjacency_multiple.data, axis=0),
        )

        median_axis1 = sim_adjacency_multiple.median(axis=1)
        assert isinstance(median_axis1, np.ndarray)
        assert len(median_axis1) == len(sim_adjacency_multiple)
        np.testing.assert_array_almost_equal(
            median_axis1, np.nanmedian(sim_adjacency_multiple.data, axis=1)
        )

    def test_sum(self):
        """Test sum handles different matrix types correctly."""
        n = 10
        a = Adjacency(np.ones((n, n)), matrix_type="directed")
        assert a.sum() == n**2
        a = Adjacency([a, a])
        assert a.sum().data.sum() == (n**2) * 2

        a = Adjacency(np.ones((n, n)), matrix_type="similarity")
        assert a.sum() == n * (n - 1) / 2
        a = Adjacency([a, a])
        assert a.sum().data.sum() == n * (n - 1)

        a = Adjacency(np.ones((n, n)), matrix_type="distance")
        assert a.sum() == n * (n - 1) / 2
        a = Adjacency([a, a])
        assert a.sum().data.sum() == n * (n - 1)

    def test_list_of_adjacency_preserves_y_and_labels(self):
        """Adjacency([adj1, adj2]) must preserve concatenated Y and labels (F032)."""
        n = 4
        labels = ["a", "b", "c", "d"]
        a1 = Adjacency(
            np.ones((n, n)),
            matrix_type="directed",
            labels=labels,
            Y=pl.DataFrame({"grp": [1]}),
        )
        a2 = Adjacency(
            np.ones((n, n)) * 2,
            matrix_type="directed",
            labels=labels,
            Y=pl.DataFrame({"grp": [2]}),
        )
        stacked = Adjacency([a1, a2])
        assert stacked.Y.shape[0] == 2
        assert stacked.Y["grp"].to_list() == [1, 2]
        assert stacked.labels == labels

    def test_directed_flat_stack_not_collapsed(self):
        """directed_flat 2-D stack must stay a stack, not collapse to 1-D (F038)."""
        n = 4
        stack = np.arange(3 * n * n, dtype=float).reshape(3, n * n)
        adj = Adjacency(stack, matrix_type="directed_flat")
        assert not adj.is_single_matrix
        assert len(adj) == 3
        assert adj.data.shape == (3, n * n)

    def test_distance(self, sim_adjacency_multiple):
        """Test distance matrix computation."""
        assert isinstance(sim_adjacency_multiple.distance(), Adjacency)
        assert sim_adjacency_multiple.distance().n_nodes == len(sim_adjacency_multiple)

    def test_similarity_conversion(self, sim_adjacency_single):
        """Test conversion between distance and similarity."""
        np.testing.assert_approx_equal(
            -1,
            pearsonr(
                sim_adjacency_single.data,
                sim_adjacency_single.distance_to_similarity().data,
            )[0],
            significant=1,
        )

    def test_distance_to_similarity_euclidean(self, sim_adjacency_single):
        """Test distance to similarity conversion using euclidean metric."""
        sim_euclidean = sim_adjacency_single.distance_to_similarity(
            metric="euclidean", beta=1
        )

        assert isinstance(sim_euclidean, Adjacency)
        assert sim_euclidean.matrix_type == "similarity"

        d = sim_adjacency_single.squareform()
        expected = np.exp(-1 * d / d.std())
        mask = ~np.eye(d.shape[0], dtype=bool)
        np.testing.assert_array_almost_equal(
            sim_euclidean.squareform()[mask], expected[mask]
        )

        sim_euclidean_beta2 = sim_adjacency_single.distance_to_similarity(
            metric="euclidean", beta=2
        )
        expected_beta2 = np.exp(-2 * d / d.std())
        np.testing.assert_array_almost_equal(
            sim_euclidean_beta2.squareform()[mask], expected_beta2[mask]
        )

        assert np.mean(sim_euclidean_beta2.data) < np.mean(sim_euclidean.data)

        with pytest.raises(ValueError, match="correlation"):
            sim_adjacency_single.distance_to_similarity(metric="invalid")
