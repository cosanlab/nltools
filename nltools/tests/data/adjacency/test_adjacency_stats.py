"""Tests for Adjacency statistical methods: similarity, NaN handling,
threshold, Fisher transforms, ttest, label distance."""

import numpy as np
import pytest

from nltools.data import Adjacency


class TestAdjacencyStats:
    @pytest.mark.slow
    def test_similarity(self, sim_adjacency_multiple):
        """Test similarity computation with permutation tests and different metrics."""
        n_permute = 100
        for metric in ["spearman", "pearson", "kendall"]:
            assert len(
                sim_adjacency_multiple.similarity(
                    sim_adjacency_multiple[0].squareform(),
                    permutation_method="1d",
                    metric=metric,
                    n_permute=n_permute,
                )
            ) == len(sim_adjacency_multiple)

        data2 = sim_adjacency_multiple[0].copy()
        rng = np.random.default_rng(seed=0)
        data2.data = data2.data + rng.standard_normal(len(data2.data)) * 0.1
        for perm_type in [None, "1d", "2d"]:
            assert (
                sim_adjacency_multiple[0].similarity(
                    data2.squareform(),
                    permutation_method=perm_type,
                    n_permute=n_permute,
                )["correlation"]
                > 0.5
            )

    @pytest.mark.slow
    def test_similarity_matrix_and_directed(self):
        """Test similarity with 2D permutation and directed matrices."""
        # Symmetric matrix permutation
        cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        dat = np.random.multivariate_normal([2, 6], cov_matrix, 190)
        x = Adjacency(dat[:, 0])
        y = Adjacency(dat[:, 1])
        stats = x.similarity(y, permutation_method="2d", n_permute=100)
        assert (
            (stats["correlation"] > 0.4)
            & (stats["correlation"] < 0.85)
            & (stats["p"] < 0.05)
        )
        stats = x.similarity(y, permutation_method=None)
        assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)

        # Directed matrices
        dat = np.random.multivariate_normal([2, 6], cov_matrix, 400)
        x = Adjacency(dat[:, 0].reshape(20, 20), matrix_type="directed")
        y = Adjacency(dat[:, 1].reshape(20, 20), matrix_type="directed")
        stats = x.similarity(
            y, permutation_method="1d", include_diag=False, n_permute=100
        )
        assert (
            (stats["correlation"] > 0.4)
            & (stats["correlation"] < 0.85)
            & (stats["p"] < 0.05)
        )
        stats = x.similarity(y, permutation_method=None, include_diag=True)
        assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)
        try:
            x.similarity(y, permutation_method="2d")
        except TypeError:
            pass

    def test_similarity_nan_handling(self):
        """Test NaN handling in similarity with all nan_policy and perm_type options."""
        rng = np.random.default_rng(42)
        cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        data = rng.multivariate_normal([2, 6], cov_matrix, 190)

        x = Adjacency(data[:, 0])
        y = Adjacency(data[:, 1])

        stats_clean = x.similarity(y, permutation_method=None, nan_policy="omit")

        x_nan = x.copy()
        y_nan = y.copy()
        x_nan.data[10] = np.nan
        y_nan.data[20] = np.nan

        # omit policy
        stats_omit = x_nan.similarity(y_nan, permutation_method=None, nan_policy="omit")
        assert not np.isnan(stats_omit["correlation"])
        assert abs(stats_omit["correlation"] - stats_clean["correlation"]) < 0.15

        # propagate policy
        stats_prop = x_nan.similarity(
            y_nan, permutation_method=None, nan_policy="propagate"
        )
        assert np.isnan(stats_prop["correlation"])

        # raise policy
        with pytest.raises(ValueError, match="Input contains NaN"):
            x_nan.similarity(y_nan, permutation_method=None, nan_policy="raise")

        # invalid policy
        with pytest.raises(ValueError, match="nan_policy must be"):
            x.similarity(y, permutation_method=None, nan_policy="invalid")

        # NaN with 1d perm_type
        n = 10
        data1 = rng.random((n, n))
        data1 = (data1 + data1.T) / 2
        data1[0, 1] = np.nan
        data1[1, 0] = np.nan
        data2 = rng.random((n, n))
        data2 = (data2 + data2.T) / 2

        adj1 = Adjacency(data1, matrix_type="similarity")
        adj2 = Adjacency(data2, matrix_type="similarity")

        result_1d = adj1.similarity(
            adj2, permutation_method="1d", n_permute=100, nan_policy="omit"
        )
        assert not np.isnan(result_1d["correlation"])
        assert "p" in result_1d

        # NaN with 2d perm_type
        with pytest.warns(UserWarning, match="NaN values detected in 2D matrix"):
            result_2d = adj1.similarity(
                adj2, permutation_method="2d", n_permute=100, nan_policy="omit"
            )
        assert "p" in result_2d
        assert 0 <= result_2d["p"] <= 1

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

    @pytest.mark.slow
    def test_ttest(self, sim_adjacency_multiple):
        """Test t-test with and without permutation."""
        out = sim_adjacency_multiple.ttest()
        assert len(out["t"]) == 1
        assert out["t"].shape[0] == sim_adjacency_multiple.shape[1]
        assert out["p"].shape[0] == sim_adjacency_multiple.shape[1]
        out = sim_adjacency_multiple.ttest(permutation=True, n_permute=100)
        assert len(out["t"]) == 1
        assert out["t"].shape[0] == sim_adjacency_multiple.shape[1]
        assert out["p"].shape[0] == sim_adjacency_multiple.shape[1]

    @pytest.mark.slow
    def test_stats_label_distance(self):
        """Test permutation tests on within and between label distances."""
        np.random.seed(42)
        n = 9
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    if labels[i] == labels[j]:
                        dist_matrix[i, j] = 0.1 + np.random.rand() * 0.1
                    else:
                        dist_matrix[i, j] = 0.8 + np.random.rand() * 0.1
        dist_matrix = (dist_matrix + dist_matrix.T) / 2

        adj = Adjacency(dist_matrix, matrix_type="distance", labels=labels)
        results = adj.stats_label_distance(labels=labels, n_permute=500)

        assert isinstance(results, dict)
        assert set(results.keys()) == {"0", "1", "2"}
        for group_key in results:
            assert "mean_diff" in results[group_key]
            assert "p" in results[group_key]
            assert results[group_key]["mean_diff"] < 0
            assert results[group_key]["p"] < 0.05

        multi_adj = Adjacency([dist_matrix, dist_matrix], matrix_type="distance")
        with pytest.raises(ValueError, match="single adjacency"):
            multi_adj.stats_label_distance(labels=labels)

        with pytest.raises(ValueError, match="same length"):
            adj.stats_label_distance(labels=np.array([0, 1]))
