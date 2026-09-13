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
                    method="1d",
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
                    method=perm_type,
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
        stats = x.similarity(y, method="2d", n_permute=100)
        assert (
            (stats["correlation"] > 0.4)
            & (stats["correlation"] < 0.85)
            & (stats["p"] < 0.05)
        )
        stats = x.similarity(y, method=None)
        assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)

        # Directed matrices
        dat = np.random.multivariate_normal([2, 6], cov_matrix, 400)
        x = Adjacency(dat[:, 0].reshape(20, 20), matrix_type="directed")
        y = Adjacency(dat[:, 1].reshape(20, 20), matrix_type="directed")
        stats = x.similarity(y, method="1d", include_diag=False, n_permute=100)
        assert (
            (stats["correlation"] > 0.4)
            & (stats["correlation"] < 0.85)
            & (stats["p"] < 0.05)
        )
        stats = x.similarity(y, method=None, include_diag=True)
        assert (stats["correlation"] > 0.4) & (stats["correlation"] < 0.85)
        try:
            x.similarity(y, method="2d")
        except TypeError:
            pass

    def test_similarity_nan_handling(self):
        """Test NaN handling in similarity with all nan_policy and perm_type options."""
        rng = np.random.default_rng(42)
        cov_matrix = np.array([[1.0, 0.7], [0.7, 1.0]])
        data = rng.multivariate_normal([2, 6], cov_matrix, 190)

        x = Adjacency(data[:, 0])
        y = Adjacency(data[:, 1])

        stats_clean = x.similarity(y, method=None, nan_policy="omit")

        x_nan = x.copy()
        y_nan = y.copy()
        x_nan.data[10] = np.nan
        y_nan.data[20] = np.nan

        # omit policy
        stats_omit = x_nan.similarity(y_nan, method=None, nan_policy="omit")
        assert not np.isnan(stats_omit["correlation"])
        assert abs(stats_omit["correlation"] - stats_clean["correlation"]) < 0.15

        # propagate policy
        stats_prop = x_nan.similarity(y_nan, method=None, nan_policy="propagate")
        assert np.isnan(stats_prop["correlation"])

        # raise policy
        with pytest.raises(ValueError, match="Input contains NaN"):
            x_nan.similarity(y_nan, method=None, nan_policy="raise")

        # invalid policy
        with pytest.raises(ValueError, match="nan_policy must be"):
            x.similarity(y, method=None, nan_policy="invalid")

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

        result_1d = adj1.similarity(adj2, method="1d", n_permute=100, nan_policy="omit")
        assert not np.isnan(result_1d["correlation"])
        assert "p" in result_1d

        # NaN with 2d perm_type: no policy makes a 2D correlation meaningful.
        for policy in ("omit", "propagate"):
            with pytest.raises(ValueError, match="method='1d'"):
                adj1.similarity(adj2, method="2d", n_permute=100, nan_policy=policy)

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

    def test_threshold_zero_cutoff(self):
        """A cutoff of 0.0 must be treated as provided, not falsy (F033)."""
        data = np.array([[0.0, -2.0, 3.0], [-2.0, 0.0, 4.0], [3.0, 4.0, 0.0]])
        adj = Adjacency(data, matrix_type="directed")
        assert (adj.data < 0).sum() == 2

        # One-sided upper=0.0 must zero everything below 0 (not be a no-op).
        thr = adj.threshold(upper=0.0)
        assert (thr.data < 0).sum() == 0

        # Two-sided lower=0.0 must engage the two-sided branch: values strictly
        # between 0.0 and 3.5 are zeroed; negatives (< lower) survive.
        thr2 = adj.threshold(lower=0.0, upper=3.5)
        assert (thr2.data == -2.0).sum() == 2

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
        n_edges = sim_adjacency_multiple.data.shape[1]
        for out in (
            sim_adjacency_multiple.ttest(),
            sim_adjacency_multiple.ttest(permutation=True, n_permute=100),
        ):
            assert set(out) == {"mean", "t", "z", "p"}
            for key in ("mean", "t", "z", "p"):
                assert len(out[key]) == 1
                assert out[key].data.shape == (n_edges,)
                assert out[key].n_nodes == sim_adjacency_multiple.n_nodes
                assert out[key].labels == sim_adjacency_multiple.labels

    def test_ttest_parametric_honors_tail(self, sim_adjacency_multiple):
        """tail=1 must reach the parametric path (was silently ignored pre-0.6.0)."""
        from scipy.stats import ttest_1samp

        out = sim_adjacency_multiple.ttest(tail=1)
        _, expected_p = ttest_1samp(
            sim_adjacency_multiple.data, 0, 0, alternative="greater"
        )
        np.testing.assert_allclose(out["p"].data, expected_p)
        with pytest.raises(ValueError, match="tail"):
            sim_adjacency_multiple.ttest(tail=-1)

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


def _adjacency_stack(
    n_matrices=8, n_nodes=4, *, matrix_type="distance", labels=None, Y=None, seed=0
):
    """Build a stack of random matrices in flat storage order."""
    rng = np.random.default_rng(seed)
    n_edges = (
        n_nodes * n_nodes if matrix_type == "directed" else n_nodes * (n_nodes - 1) // 2
    )
    values = rng.standard_normal((n_matrices, n_edges)) + 0.3
    return Adjacency(values, matrix_type=f"{matrix_type}_flat", labels=labels, Y=Y)


class TestAdjacencyTTest:
    """Shared one-sample t-test contract (docs/development/specs/ttest.md)."""

    def test_ttest_permutation_matches_engine_at_fixed_seed(self):
        from scipy.stats import ttest_1samp

        from nltools.algorithms.inference import one_sample_permutation_test

        stack = _adjacency_stack()
        popmean = 0.2
        out = stack.ttest(
            popmean=popmean,
            permutation=True,
            n_permute=64,
            return_null=True,
            random_state=11,
        )
        engine = one_sample_permutation_test(
            stack.data - popmean,
            n_permute=64,
            tail=2,
            return_null=True,
            n_jobs=-1,
            random_state=11,
        )
        np.testing.assert_allclose(out["p"].data, engine["p"])
        np.testing.assert_allclose(out["mean"].data, engine["mean"])
        np.testing.assert_allclose(out["null_dist"], engine["null_dist"])
        assert out["null_dist"].shape == (64, stack.data.shape[1])
        # t is the observed statistic, not the mean the null holds.
        expected_t, _ = ttest_1samp(stack.data, popmean, axis=0)
        np.testing.assert_allclose(out["t"].data, expected_t)

    def test_ttest_null_only_when_permuting_and_requested(self):
        stack = _adjacency_stack()
        assert "null_dist" not in stack.ttest(return_null=True)
        assert "null_dist" not in stack.ttest(
            permutation=True, n_permute=16, random_state=0
        )
        with_null = stack.ttest(
            permutation=True, n_permute=16, return_null=True, random_state=0
        )
        without_null = stack.ttest(
            permutation=True, n_permute=16, return_null=False, random_state=0
        )
        for key in ("mean", "t", "z", "p"):
            np.testing.assert_array_equal(with_null[key].data, without_null[key].data)

    def test_ttest_z_follows_the_shared_conversion(self):
        from scipy.stats import norm

        stack = _adjacency_stack()
        two = stack.ttest()
        np.testing.assert_allclose(
            two["z"].data,
            np.sign(two["t"].data) * norm.isf(np.asarray(two["p"].data) / 2.0),
        )
        upper = stack.ttest(tail=1)
        np.testing.assert_allclose(
            upper["z"].data, norm.isf(np.asarray(upper["p"].data))
        )

    def test_ttest_symmetric_result_has_a_zero_diagonal(self):
        stack = _adjacency_stack()
        square = stack.ttest()["t"].squareform()
        assert square.shape == (stack.n_nodes, stack.n_nodes)
        np.testing.assert_array_equal(np.diag(square), np.zeros(stack.n_nodes))

    def test_ttest_directed_storage_and_node_order_survive(self):
        labels = ["a", "b", "c"]
        stack = _adjacency_stack(n_nodes=3, matrix_type="directed", labels=labels)
        out = stack.ttest()
        for key in ("mean", "t", "z", "p"):
            assert out[key].matrix_type == "directed"
            assert out[key].n_nodes == 3
            assert out[key].labels == labels
            assert np.asarray(out[key].data).shape == (9,)
        # Flat storage order is preserved: t reshapes to the directed square.
        np.testing.assert_allclose(
            out["t"].squareform(), np.asarray(out["t"].data).reshape(3, 3)
        )

    def test_ttest_retains_shared_and_consistent_labels(self):
        labels = ["a", "b", "c", "d"]
        shared = _adjacency_stack(labels=labels)
        assert shared.ttest()["t"].labels == labels
        nested = _adjacency_stack(n_matrices=3, labels=[labels] * 3)
        assert nested.ttest()["t"].labels == labels

    def test_ttest_clears_inconsistent_labels_and_matrix_metadata(self):
        import polars as pl

        labels = [
            ["a", "b", "c", "d"],
            ["a", "b", "c", "d"],
            ["w", "x", "y", "z"],
        ]
        stack = _adjacency_stack(
            n_matrices=3,
            labels=labels,
            Y=pl.DataFrame({"group": [0, 1, 1]}),
        )
        out = stack.ttest()
        for key in ("mean", "t", "z", "p"):
            assert out[key].labels == []
            assert out[key].Y.is_empty()

    def test_ttest_requires_two_matrices(self):
        single = _adjacency_stack(n_matrices=1)[0]
        with pytest.raises(ValueError, match="multiple matrices"):
            single.ttest()
        one_row_stack = _adjacency_stack(n_matrices=1)
        assert not one_row_stack.is_single_matrix
        with pytest.raises(ValueError, match="multiple matrices"):
            one_row_stack.ttest()

    def test_ttest_results_are_owned(self):
        stack = _adjacency_stack()
        original = stack.data.copy()
        out = stack.ttest(
            permutation=True, n_permute=16, return_null=True, random_state=0
        )
        arrays = [out[key].data for key in ("mean", "t", "z", "p")]
        arrays.append(out["null_dist"])
        for i, first in enumerate(arrays):
            assert not np.shares_memory(first, stack.data)
            for second in arrays[i + 1 :]:
                assert not np.shares_memory(first, second)
        for array in arrays:
            array[...] = -999.0
        np.testing.assert_array_equal(stack.data, original)

    def test_ttest_maps_are_unthresholded_and_threshold_applies_afterwards(self):
        stack = _adjacency_stack()
        out = stack.ttest()
        p_values = np.asarray(out["p"].data)
        assert np.any(p_values > 0.05)
        assert np.all(np.asarray(out["t"].data) != 0)

        significant = out["t"].copy()
        significant.data = np.where(p_values < 0.05, significant.data, 0.0)
        assert np.all(significant.data[p_values >= 0.05] == 0)
        np.testing.assert_allclose(
            significant.data[p_values < 0.05],
            np.asarray(out["t"].data)[p_values < 0.05],
        )
