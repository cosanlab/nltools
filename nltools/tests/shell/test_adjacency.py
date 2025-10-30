"""
Test suite for Adjacency class.

Follows "imperative shell" pattern: tests focus on matrix type handling,
operations, and interface contracts. Organized into logical sections for clarity.
"""

import os
import pytest
import numpy as np
import pandas as pd
from nltools.data import Adjacency, Design_Matrix
import networkx as nx
from scipy.stats import pearsonr
from scipy.linalg import block_diag
from pathlib import Path
from sklearn.metrics import pairwise_distances


class TestAdjacency:
    """Test Adjacency class - focus on matrix operations and type handling."""

    # ==================== Initialization & Type Inference ====================

    def test_type_single(self, sim_adjacency_single):
        """Test symmetric matrix type detection (distance vs similarity)."""
        assert sim_adjacency_single.matrix_type == "distance"
        dat_single2 = Adjacency(1 - sim_adjacency_single.squareform())
        assert dat_single2.matrix_type == "similarity"
        assert sim_adjacency_single.issymmetric

    def test_type_directed(self, sim_adjacency_directed):
        """Test directed matrix initialization."""
        assert not sim_adjacency_directed.issymmetric

    # ==================== Basic Properties & Indexing ====================

    def test_length(self, sim_adjacency_multiple):
        """Test length property for multiple adjacency matrices."""
        assert len(sim_adjacency_multiple) == sim_adjacency_multiple.data.shape[0]
        assert len(sim_adjacency_multiple[0]) == 1

    def test_indexing(self, sim_adjacency_multiple):
        """Test indexing with integers, slices, and tuples."""
        assert len(sim_adjacency_multiple[0]) == 1
        assert len(sim_adjacency_multiple[0:4]) == 4
        assert len(sim_adjacency_multiple[0, 2, 3]) == 3

    # ==================== Arithmetic Operations ====================

    def test_arithmetic(self, sim_adjacency_directed):
        """Test arithmetic operations on adjacency matrices."""
        assert (sim_adjacency_directed + 5).data[0] == sim_adjacency_directed.data[0] + 5
        assert (sim_adjacency_directed - 0.5).data[0] == sim_adjacency_directed.data[
            0
        ] - 0.5
        assert (sim_adjacency_directed * 5).data[0] == sim_adjacency_directed.data[0] * 5
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
            ((2 * sim_adjacency_directed / 2) / sim_adjacency_directed).mean(), 1, decimal=4
        )

    def test_copy(self, sim_adjacency_multiple):
        """Test copying adjacency matrices."""
        assert np.all(sim_adjacency_multiple.data == sim_adjacency_multiple.copy().data)

    # ==================== Matrix Operations ====================

    def test_squareform(self, sim_adjacency_multiple):
        """Test vector → matrix → vector conversion preserves data."""
        assert len(sim_adjacency_multiple.squareform()) == len(sim_adjacency_multiple)
        assert (
            sim_adjacency_multiple[0].squareform().shape
            == sim_adjacency_multiple[0].square_shape()
        )

    def test_append(self, sim_adjacency_single):
        """Test appending adjacency matrices."""
        a = Adjacency()
        a = a.append(sim_adjacency_single)
        assert a.shape == sim_adjacency_single.shape
        a = a.append(a)
        assert a.shape == (2, 6)

    # ==================== I/O Operations ====================

    def test_write_multiple(self, sim_adjacency_multiple, tmpdir):
        """Test writing and loading multiple adjacency matrices (CSV and HDF5)."""
        sim_adjacency_multiple.write(
            os.path.join(str(tmpdir.join("Test.csv"))), method="long"
        )
        dat_multiple2 = Adjacency(
            os.path.join(str(tmpdir.join("Test.csv"))), matrix_type="distance_flat"
        )
        assert np.all(np.isclose(sim_adjacency_multiple.data, dat_multiple2.data))

        # Test i/o for hdf5
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

    def test_load_legacy_h5(
        self, old_h5_adj_single, new_h5_adj_single, old_h5_adj_double, new_h5_adj_double, tmpdir
    ):
        """Test loading old HDF5 format (backward compatibility)."""
        with pytest.warns(UserWarning):
            # With verbosity on we should see a warning about the old h5 file format
            b_old = Adjacency(old_h5_adj_single, verbose=True)
        b_new = Adjacency(new_h5_adj_single)
        assert b_old.shape == b_new.shape
        assert np.allclose(b_old.data, b_new.data)
        # NOTE: We lose pandas column dtype information between old and new h5 files
        # so we can't use .equals()
        assert b_old.Y.shape == b_new.Y.shape
        assert b_old.matrix_type == b_new.matrix_type
        assert b_old.is_single_matrix == b_new.is_single_matrix
        assert b_old.issymmetric == b_new.issymmetric
        assert b_old.labels == b_new.labels

        b_old = Adjacency(old_h5_adj_double, legacy_h5=True)
        b_new = Adjacency(new_h5_adj_double)
        assert b_old.shape == b_new.shape
        assert np.allclose(b_old.data, b_new.data)
        # NOTE: We lose pandas column dtype information between old and new h5 files
        # so we can't use .equals()
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

    # ==================== Aggregation Methods ====================

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

    # ==================== Similarity & Distance Methods ====================

    @pytest.mark.tier2
    def test_similarity(self, sim_adjacency_multiple):
        """Test similarity computation with permutation tests and different metrics."""
        n_permute = 1000
        assert len(
            sim_adjacency_multiple.similarity(
                sim_adjacency_multiple[0].squareform(), perm_type="1d", n_permute=n_permute
            )
        ) == len(sim_adjacency_multiple)
        assert len(
            sim_adjacency_multiple.similarity(
                sim_adjacency_multiple[0].squareform(),
                perm_type="1d",
                metric="pearson",
                n_permute=n_permute,
            )
        ) == len(sim_adjacency_multiple)
        assert len(
            sim_adjacency_multiple.similarity(
                sim_adjacency_multiple[0].squareform(),
                perm_type="1d",
                metric="kendall",
                n_permute=n_permute,
            )
        ) == len(sim_adjacency_multiple)

        data2 = sim_adjacency_multiple[0].copy()
        data2.data = data2.data + np.random.randn(len(data2.data)) * 0.1
        assert (
            sim_adjacency_multiple[0].similarity(
                data2.squareform(), perm_type=None, n_permute=n_permute
            )["correlation"]
            > 0.5
        )
        assert (
            sim_adjacency_multiple[0].similarity(
                data2.squareform(), perm_type="1d", n_permute=n_permute
            )["correlation"]
            > 0.5
        )
        assert (
            sim_adjacency_multiple[0].similarity(
                data2.squareform(), perm_type="2d", n_permute=n_permute
            )["correlation"]
            > 0.5
        )

    def test_similarity_matrix_permutation(self):
        """Test similarity with 2D matrix permutation."""
        # Create a positive definite covariance matrix
        cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        dat = np.random.multivariate_normal([2, 6], cov_matrix, 190)
        x = Adjacency(dat[:, 0])
        y = Adjacency(dat[:, 1])
        stats = x.similarity(y, perm_type="2d", n_permute=1000)
        assert (
            (stats["correlation"] > 0.4)
            & (stats["correlation"] < 0.85)
            & (stats["p"] < 0.001)
        )
        stats = x.similarity(y, perm_type=None)
        assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)

    def test_directed_similarity(self):
        """Test similarity for directed matrices."""
        # Create a positive definite covariance matrix
        cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        dat = np.random.multivariate_normal([2, 6], cov_matrix, 400)
        x = Adjacency(dat[:, 0].reshape(20, 20), matrix_type="directed")
        y = Adjacency(dat[:, 1].reshape(20, 20), matrix_type="directed")
        # Ignore diagonal
        stats = x.similarity(y, perm_type="1d", ignore_diagonal=True, n_permute=1000)
        assert (
            (stats["correlation"] > 0.4)
            & (stats["correlation"] < 0.85)
            & (stats["p"] < 0.001)
        )
        # Use diagonal
        stats = x.similarity(y, perm_type=None, ignore_diagonal=False)
        assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)
        # Error out but make sure TypeError is the reason why
        try:
            x.similarity(y, perm_type="2d")
        except TypeError as _:  # noqa
            pass

    def test_distance(self, sim_adjacency_multiple):
        """Test distance matrix computation."""
        assert isinstance(sim_adjacency_multiple.distance(), Adjacency)
        assert sim_adjacency_multiple.distance().square_shape()[0] == len(
            sim_adjacency_multiple
        )

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

    # ==================== Statistical Methods ====================

    def test_ttest(self, sim_adjacency_multiple):
        """Test t-test with and without permutation."""
        out = sim_adjacency_multiple.ttest()
        assert len(out["t"]) == 1
        assert len(out["p"]) == 1
        assert out["t"].shape[0] == sim_adjacency_multiple.shape[1]
        assert out["p"].shape[0] == sim_adjacency_multiple.shape[1]
        out = sim_adjacency_multiple.ttest(permutation=True, n_permute=1000)
        assert len(out["t"]) == 1
        assert len(out["p"]) == 1
        assert out["t"].shape[0] == sim_adjacency_multiple.shape[1]
        assert out["p"].shape[0] == sim_adjacency_multiple.shape[1]

    def test_bootstrap(self, sim_adjacency_multiple):
        """Test bootstrap resampling."""
        n_samples = 3
        b = sim_adjacency_multiple.bootstrap("mean", n_samples=n_samples)
        assert isinstance(b["Z"], Adjacency)
        b = sim_adjacency_multiple.bootstrap("std", n_samples=n_samples)
        assert isinstance(b["Z"], Adjacency)

    def test_isc(self, sim_adjacency_single):
        """Test intersubject correlation."""
        n_boot = 100
        for metric in ["median", "mean"]:
            stats = sim_adjacency_single.isc(
                metric=metric, n_samples=n_boot, return_null=True
            )
            assert (stats["isc"] > -1) & (stats["isc"] < 1)
            assert (stats["p"] > 0) & (stats["p"] < 1)
            assert len(stats["null_distribution"]) == n_boot

    def test_isc_group_adj(self):
        """Test group-level ISC comparison."""
        n_samples = 100
        diff = 0.2
        data = np.random.multivariate_normal(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                [1, 0.2, 0.5, 0.7, 0.3, 0, 0, 0, 0, 0],
                [0.2, 1, 0.6, 0.1, 0.2, 0, 0, 0, 0, 0],
                [0.5, 0.6, 1, 0.3, 0.1, 0, 0, 0, 0, 0],
                [0.7, 0.1, 0.3, 1, 0.4, 0, 0, 0, 0, 0],
                [0.3, 0.2, 0.1, 0.4, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0.2 + diff, 0.5 + diff, 0.7 + diff, 0.3 + diff],
                [0, 0, 0, 0, 0, 0.2 + diff, 1, 0.6 + diff, 0.1 + diff, 0.2 + diff],
                [0, 0, 0, 0, 0, 0.5 + diff, 0.6 + diff, 1, 0.3 + diff, 0.1 + diff],
                [0, 0, 0, 0, 0, 0.7 + diff, 0.1 + diff, 0.3 + diff, 1, 0.4 + diff],
                [0, 0, 0, 0, 0, 0.3 + diff, 0.2 + diff, 0.1 + diff, 0.4 + diff, 1],
            ],
            500,
        )

        group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        similarity = Adjacency(
            1 - pairwise_distances(data.T, metric="correlation"), matrix_type="similarity"
        )

        for method in ["permute", "bootstrap"]:
            for metric in ["median", "mean"]:
                stats = similarity.isc_group(
                    group,
                    metric=metric,
                    method=method,
                    return_null=True,
                    n_samples=n_samples,
                )
                np.testing.assert_almost_equal(
                    stats["isc_group_difference"], diff, decimal=0
                )
                assert (stats["p"] > 0) & (stats["p"] < 1)
                assert len(stats["null_distribution"]) == n_samples

    # ==================== Transform & Utility Methods ====================

    def test_threshold(self, sim_adjacency_directed):
        """Test thresholding matrices."""
        assert np.sum(sim_adjacency_directed.threshold(upper=0.8).data == 0) == 10
        assert sim_adjacency_directed.threshold(upper=0.8, binarize=True).data[0]
        assert (
            np.sum(sim_adjacency_directed.threshold(upper="70%", binarize=True).data) == 5
        )
        assert np.sum(sim_adjacency_directed.threshold(lower=0.4, binarize=True).data) == 6

    def test_fisher_r_to_z(self, sim_adjacency_single):
        """Test Fisher r-to-z transformation."""
        np.testing.assert_almost_equal(
            np.nansum(
                sim_adjacency_single.data - sim_adjacency_single.r_to_z().z_to_r().data
            ),
            0,
            decimal=2,
        )

    # ==================== Graph Operations ====================

    def test_graph_directed(self, sim_adjacency_directed):
        """Test conversion to directed graph."""
        assert isinstance(sim_adjacency_directed.to_graph(), nx.DiGraph)

    def test_graph_single(self, sim_adjacency_single):
        """Test conversion to undirected graph."""
        assert isinstance(sim_adjacency_single.to_graph(), nx.Graph)

    # ==================== Regression & Analysis ====================

    @pytest.mark.skip(
        reason="Adjacency.regress() needs refactoring for Polars DesignMatrix. "
        "Requires: .T attribute and numpy interop. "
        "Defer to v0.6.1+ when Adjacency module is refactored."
    )
    def test_regression(self):
        """Test regression with Adjacency and Design_Matrix predictors."""
        # Test Adjacency Regression
        m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
        m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
        Y = Adjacency(m1 * 1 + m2 * 2 + m3 * 3, matrix_type="similarity")
        X = Adjacency([m1, m2, m3], matrix_type="similarity")

        stats = Y.regress(X)
        assert np.allclose(stats["beta"], np.array([1, 2, 3]))

        # Test Design_Matrix Regression
        n = 10
        d = Adjacency(
            [
                block_diag(np.ones((4, 4)) + np.random.randn(4, 4) * 0.1, np.zeros((8, 8)))
                for _ in range(n)
            ],
            matrix_type="similarity",
        )
        X = Design_Matrix(np.ones(n))
        stats = d.regress(X)
        out = stats["beta"].cluster_summary(
            clusters=["Group1"] * 4 + ["Group2"] * 8, summary="within"
        )
        assert np.allclose(
            np.array([out["Group1"], out["Group2"]]), np.array([1, 0]), rtol=1e-01
        )

    def test_social_relations_model(self):
        """Test Social Relations Model (SRM) analysis."""
        data = Adjacency(
            np.array(
                [
                    [np.nan, 8, 5, 10],
                    [7, np.nan, 7, 6],
                    [8, 7, np.nan, 5],
                    [4, 5, 0, np.nan],
                ]
            ),
            matrix_type="directed",
        )
        data2 = data.append(data)
        results1 = data.social_relations_model()
        assert isinstance(data.social_relations_model(), pd.Series)
        assert isinstance(data2.social_relations_model(), pd.DataFrame)
        assert len(results1["actor_effect"]) == data.square_shape()[0]
        assert results1["relationship_effect"].shape == data.square_shape()
        np.testing.assert_approx_equal(results1["actor_variance"], 3.33, significant=2)
        np.testing.assert_approx_equal(results1["partner_variance"], 0.66, significant=2)
        np.testing.assert_approx_equal(
            results1["relationship_variance"], 3.33, significant=2
        )
        np.testing.assert_approx_equal(
            results1["actor_partner_correlation"], 0.22, significant=2
        )
        np.testing.assert_approx_equal(
            results1["dyadic_reciprocity_correlation"], 0.2, significant=2
        )

    def test_cluster_summary(self):
        """Test cluster-based summary statistics."""
        m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
        m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
        noisy = (m1 * 1 + m2 * 2 + m3 * 3) + np.random.randn(12, 12) * 0.1
        dat = Adjacency(
            noisy, matrix_type="similarity", labels=["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4
        )

        clusters = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        cluster_mean = dat.cluster_summary(clusters=clusters)
        for i, j in zip(
            np.array([1, 2, 3]), np.array([cluster_mean[x] for x in cluster_mean])
        ):
            np.testing.assert_almost_equal(i, j, decimal=1)

        for i in dat.cluster_summary(clusters=clusters, summary="between").values():
            np.testing.assert_almost_equal(0, i, decimal=1)

        for i in dat.cluster_summary(clusters=clusters, summary="between").values():
            np.testing.assert_almost_equal(0, i, decimal=1)
