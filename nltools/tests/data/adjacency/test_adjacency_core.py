"""Tests for Adjacency core operations: init, type inference, properties,
indexing, arithmetic, copy, squareform, append, aggregation, distance."""

import numpy as np
import polars as pl
import pytest
from scipy.spatial.distance import correlation
from scipy.stats import pearsonr

from nltools.data import Adjacency


class TestAdjacencyCore:
    def test_length(self, sim_adjacency_multiple):
        """Test length property for multiple adjacency matrices."""
        assert len(sim_adjacency_multiple) == sim_adjacency_multiple.data.shape[0]
        assert len(sim_adjacency_multiple[0]) == 1

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

    def test_distance_include_diag_keeps_main_diagonal(self):
        """#384: `include_diag=True` compares upper triangles that carry the diagonal.

        Off the diagonal the two matrices are perfectly correlated; only the
        stored zero diagonal separates them.
        """
        first = np.array([[0.0, 1, 2], [1, 0, 3], [2, 3, 0]])
        second = np.array([[0.0, 4, 5], [4, 0, 6], [5, 6, 0]])
        stack = Adjacency([first, second])

        expected = correlation([0.0, 1, 2, 0, 3, 0], [0.0, 4, 5, 0, 6, 0])
        with_diag = stack.distance(metric="correlation", include_diag=True)
        without_diag = stack.distance(metric="correlation")

        assert with_diag.squareform()[0, 1] == pytest.approx(expected)
        assert without_diag.squareform()[0, 1] == pytest.approx(0.0, abs=1e-12)

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
