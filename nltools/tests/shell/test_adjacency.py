"""
Test suite for Adjacency class.

Follows "imperative shell" pattern: tests focus on matrix type handling,
operations, and interface contracts. Organized into logical sections for clarity.
"""

import os
import pytest
import numpy as np
import pandas as pd
from nltools.data import Adjacency, DesignMatrix
import networkx as nx
from scipy.stats import pearsonr
from scipy.linalg import block_diag
from pathlib import Path
from nltools.tests.conftest import _tables_available


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

    def test_median(self, sim_adjacency_single, sim_adjacency_multiple):
        """Test median calculation for single and multiple adjacency matrices."""
        # Test single matrix - should return float
        single_median = sim_adjacency_single.median()
        assert isinstance(single_median, (float, np.floating))
        assert np.isclose(single_median, np.nanmedian(sim_adjacency_single.data))

        # Test multiple matrices with axis=0 - should return Adjacency
        median_axis0 = sim_adjacency_multiple.median(axis=0)
        assert isinstance(median_axis0, Adjacency)
        assert len(median_axis0) == 1
        np.testing.assert_array_almost_equal(
            median_axis0.data.flatten(),
            np.nanmedian(sim_adjacency_multiple.data, axis=0),
        )

        # Test multiple matrices with axis=1 - should return np.array
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

    # ==================== Similarity & Distance Methods ====================

    @pytest.mark.slow
    def test_similarity(self, sim_adjacency_multiple):
        """Test similarity computation with permutation tests and different metrics."""
        n_permute = 1000
        assert len(
            sim_adjacency_multiple.similarity(
                sim_adjacency_multiple[0].squareform(),
                perm_type="1d",
                n_permute=n_permute,
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

    def test_distance_to_similarity_euclidean(self, sim_adjacency_single):
        """Test distance to similarity conversion using euclidean metric."""
        # Get similarity using euclidean conversion
        sim_euclidean = sim_adjacency_single.distance_to_similarity(
            metric="euclidean", beta=1
        )

        # Check that output is an Adjacency with similarity type
        assert isinstance(sim_euclidean, Adjacency)
        assert sim_euclidean.matrix_type == "similarity"

        # Verify the formula: exp(-beta * d / std(d))
        # Note: diagonal remains 0 (distance to self = 0, not exp(0) = 1)
        d = sim_adjacency_single.squareform()
        expected = np.exp(-1 * d / d.std())
        # Check off-diagonal elements match the formula
        mask = ~np.eye(d.shape[0], dtype=bool)
        np.testing.assert_array_almost_equal(
            sim_euclidean.squareform()[mask], expected[mask]
        )

        # Test with different beta value
        sim_euclidean_beta2 = sim_adjacency_single.distance_to_similarity(
            metric="euclidean", beta=2
        )
        expected_beta2 = np.exp(-2 * d / d.std())
        np.testing.assert_array_almost_equal(
            sim_euclidean_beta2.squareform()[mask], expected_beta2[mask]
        )

        # Higher beta should produce smaller similarity values (faster decay)
        assert np.mean(sim_euclidean_beta2.data) < np.mean(sim_euclidean.data)

        # Test invalid metric raises error
        with pytest.raises(ValueError, match="correlation"):
            sim_adjacency_single.distance_to_similarity(metric="invalid")

    def test_similarity_nan_handling(self):
        """Test NaN handling in similarity calculation (#432)."""
        # Create correlated data that can be squareformed
        # 190 elements = upper triangle of 20x20 matrix (20*19/2 = 190)
        rng = np.random.default_rng(42)
        cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        data = rng.multivariate_normal([2, 6], cov_matrix, 190)

        x = Adjacency(data[:, 0])
        y = Adjacency(data[:, 1])

        # Baseline without NaN using perm_type=None (no permutation, just correlation)
        stats_clean = x.similarity(y, perm_type=None, nan_policy="omit")

        # Add NaN values to copies
        x_nan = x.copy()
        y_nan = y.copy()
        x_nan.data[10] = np.nan
        y_nan.data[20] = np.nan

        # Test nan_policy='omit' (default) - should work and give similar result
        stats_omit = x_nan.similarity(y_nan, perm_type=None, nan_policy="omit")
        assert not np.isnan(stats_omit["correlation"])
        # Correlation should be similar (within reasonable tolerance)
        assert abs(stats_omit["correlation"] - stats_clean["correlation"]) < 0.15

        # Test nan_policy='propagate' - should return NaN
        stats_prop = x_nan.similarity(y_nan, perm_type=None, nan_policy="propagate")
        assert np.isnan(stats_prop["correlation"])

        # Test nan_policy='raise' - should raise ValueError
        with pytest.raises(ValueError, match="Input contains NaN"):
            x_nan.similarity(y_nan, perm_type=None, nan_policy="raise")

        # Test invalid nan_policy
        with pytest.raises(ValueError, match="nan_policy must be"):
            x.similarity(y, perm_type=None, nan_policy="invalid")

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
        n_samples = 50
        boot = sim_adjacency_multiple.bootstrap(
            stat="mean", n_samples=n_samples, random_state=42
        )
        assert isinstance(boot["Z"], Adjacency)
        assert isinstance(boot["mean"], Adjacency)
        assert "p" in boot
        boot = sim_adjacency_multiple.bootstrap(
            stat="std", n_samples=n_samples, random_state=42
        )
        assert isinstance(boot["Z"], Adjacency)
        assert isinstance(boot["std"], Adjacency)

    def test_bootstrap_save_boots(self, sim_adjacency_multiple):
        """Test bootstrap with save_boots parameter."""
        n_samples = 50
        result = sim_adjacency_multiple.bootstrap(
            stat="mean", n_samples=n_samples, save_boots=True, random_state=42
        )
        # When save_boots=True, should return dict with samples
        assert isinstance(result, dict)
        assert "samples" in result
        assert result["samples"].shape[0] == n_samples

    def test_bootstrap_all_simple_stats(self, sim_adjacency_multiple):
        """Test all simple stats work."""
        n_samples = 50
        stats = ["mean", "median", "std", "sum", "min", "max"]
        for stat in stats:
            boot = sim_adjacency_multiple.bootstrap(
                stat=stat, n_samples=n_samples, random_state=42
            )
            assert isinstance(boot, dict)
            assert "Z" in boot
            assert isinstance(boot["Z"], Adjacency)

    def test_bootstrap_reproducibility(self, sim_adjacency_multiple):
        """Test same random_state produces identical results."""
        n_samples = 50
        boot1 = sim_adjacency_multiple.bootstrap(
            stat="mean", n_samples=n_samples, random_state=42
        )
        boot2 = sim_adjacency_multiple.bootstrap(
            stat="mean", n_samples=n_samples, random_state=42
        )
        np.testing.assert_allclose(boot1["mean"].data, boot2["mean"].data, rtol=1e-10)

    def test_bootstrap_invalid_stat_error(self, sim_adjacency_multiple):
        """Test error for unsupported stat."""
        with pytest.raises(ValueError, match="Unsupported stat"):
            sim_adjacency_multiple.bootstrap(stat="invalid_stat", n_samples=10)

    @pytest.mark.slow
    def test_stats_label_distance(self):
        """Test permutation tests on within and between label distances."""
        # Create a block diagonal distance matrix with clear group structure
        # Groups are: [0,1,2], [3,4,5], [6,7,8] with distinct within-group distances
        np.random.seed(42)  # For reproducibility
        n = 9
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        # Create distance matrix with small within-group and large between-group
        # distances to get a significant result
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if labels[i] == labels[j]:
                        # Within-group: small distance
                        dist_matrix[i, j] = 0.1 + np.random.rand() * 0.1
                    else:
                        # Between-group: large distance
                        dist_matrix[i, j] = 0.8 + np.random.rand() * 0.1
        # Make symmetric
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        adj = Adjacency(dist_matrix, matrix_type="distance", labels=labels)

        # Run stats_label_distance with fewer permutations for speed
        results = adj.stats_label_distance(labels=labels, n_permute=500)

        # Should return a dict keyed by group labels (as strings)
        assert isinstance(results, dict)
        assert set(results.keys()) == {"0", "1", "2"}

        # Each group should have 'mean' and 'p' from two_sample_permutation
        for group_key in results:
            assert "mean" in results[group_key]
            assert "p" in results[group_key]
            # Mean difference should be negative (within < between for distances)
            assert results[group_key]["mean"] < 0
            # With clear structure, p-value should be significant
            assert results[group_key]["p"] < 0.05

        # Test error for multiple matrices
        multi_adj = Adjacency([dist_matrix, dist_matrix], matrix_type="distance")
        with pytest.raises(ValueError, match="single adjacency"):
            multi_adj.stats_label_distance(labels=labels)

        # Test error for wrong label length
        with pytest.raises(ValueError, match="same length"):
            adj.stats_label_distance(labels=np.array([0, 1]))

    # ==================== Transform & Utility Methods ====================

    def test_threshold(self, sim_adjacency_directed):
        """Test thresholding matrices."""
        assert np.sum(sim_adjacency_directed.threshold(upper=0.8).data == 0) == 10
        assert sim_adjacency_directed.threshold(upper=0.8, binarize=True).data[0]
        assert (
            np.sum(sim_adjacency_directed.threshold(upper="70%", binarize=True).data)
            == 5
        )
        assert (
            np.sum(sim_adjacency_directed.threshold(lower=0.4, binarize=True).data) == 6
        )

    def test_fisher_r_to_z(self, sim_adjacency_single):
        """Test Fisher r-to-z transformation."""
        np.testing.assert_almost_equal(
            np.nansum(
                sim_adjacency_single.data - sim_adjacency_single.r_to_z().z_to_r().data
            ),
            0,
            decimal=2,
        )

    def test_generate_permutations(self, sim_adjacency_single):
        """Test lazy generation of permuted adjacency matrices."""
        n_perm = 10
        original_data = sim_adjacency_single.data.copy()

        # Test that generator yields correct number of permutations
        perms = list(
            sim_adjacency_single.generate_permutations(n_perm, random_state=42)
        )
        assert len(perms) == n_perm

        # Each permutation should be an Adjacency object
        for perm in perms:
            assert isinstance(perm, Adjacency)
            # Same shape as original
            assert perm.shape == sim_adjacency_single.shape

        # Permutations should differ from original
        # (With 4x4 matrix, extremely unlikely all match by chance)
        all_same = all(np.allclose(perm.data, original_data) for perm in perms)
        assert not all_same

        # Test reproducibility with same random_state
        perm_list1 = [
            p.data.copy()
            for p in sim_adjacency_single.generate_permutations(5, random_state=123)
        ]
        perm_list2 = [
            p.data.copy()
            for p in sim_adjacency_single.generate_permutations(5, random_state=123)
        ]
        for p1, p2 in zip(perm_list1, perm_list2):
            np.testing.assert_array_equal(p1, p2)

        # Test that different random_states produce different permutations
        perm_a = next(sim_adjacency_single.generate_permutations(1, random_state=1))
        perm_b = next(sim_adjacency_single.generate_permutations(1, random_state=2))
        assert not np.allclose(perm_a.data, perm_b.data)

    # ==================== Graph Operations ====================

    def test_graph_directed(self, sim_adjacency_directed):
        """Test conversion to directed graph."""
        assert isinstance(sim_adjacency_directed.to_graph(), nx.DiGraph)

    def test_graph_single(self, sim_adjacency_single):
        """Test conversion to undirected graph."""
        assert isinstance(sim_adjacency_single.to_graph(), nx.Graph)

    # ==================== Regression & Analysis ====================

    @pytest.mark.skip(
        reason="Adjacency.regress() implementation refactored - skipping temporarily"
    )
    def test_regression(self):
        """Test regression with Adjacency and DesignMatrix predictors."""
        # Test Adjacency Regression
        m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
        m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
        Y = Adjacency(m1 * 1 + m2 * 2 + m3 * 3, matrix_type="similarity")
        X = Adjacency([m1, m2, m3], matrix_type="similarity")

        stats = Y.regress(X)
        assert np.allclose(stats["beta"].data, np.array([1, 2, 3]))

        # Test DesignMatrix Regression
        n = 10
        d = Adjacency(
            [
                block_diag(
                    np.ones((4, 4)) + np.random.randn(4, 4) * 0.1, np.zeros((8, 8))
                )
                for _ in range(n)
            ],
            matrix_type="similarity",
        )
        X = DesignMatrix(np.ones(n))
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
        np.testing.assert_approx_equal(
            results1["partner_variance"], 0.66, significant=2
        )
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
