"""Tests for Adjacency modeling: bootstrap, regression, SRM, cluster summary,
permutation generation."""

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from nltools.data import Adjacency, DesignMatrix

# Only `bootstrap` emits these two quality advisories, and the bootstrap tests
# below run deliberately small resamples that trip them by design. Filtering
# them here keeps the suite's warning count from tracking the number of
# bootstrap call sites; `pytest.warns` still sees them where one is asserted.
pytestmark = [
    pytest.mark.filterwarnings("ignore:n_samples=:UserWarning"),
    pytest.mark.filterwarnings("ignore:Only .* samples available:UserWarning"),
]


class TestAdjacencyModeling:
    @pytest.mark.slow
    def test_bootstrap(self, sim_adjacency_multiple):
        """Test bootstrap resampling."""
        from nltools.data.results import BootstrapResult

        n_samples = 50
        boot = sim_adjacency_multiple.bootstrap(
            "mean", n_samples=n_samples, random_state=42
        )
        assert isinstance(boot, BootstrapResult)
        for field in ("estimate", "standard_error", "ci_lower", "ci_upper"):
            assert isinstance(getattr(boot, field), Adjacency)
        assert boot.samples is None

    @pytest.mark.slow
    def test_bootstrap_estimate_is_the_unresampled_statistic(
        self, sim_adjacency_multiple
    ):
        boot = sim_adjacency_multiple.bootstrap("mean", n_samples=50, random_state=42)

        np.testing.assert_allclose(
            boot.estimate.data,
            np.mean(sim_adjacency_multiple.data.astype(np.float64), axis=0),
            rtol=1e-12,
        )

    @pytest.mark.slow
    def test_bootstrap_return_samples(self, sim_adjacency_multiple):
        """Retained replicates place the bootstrap axis first."""
        n_samples = 50
        result = sim_adjacency_multiple.bootstrap(
            "mean", n_samples=n_samples, return_samples=True, random_state=42
        )

        assert result.samples.shape[0] == n_samples
        np.testing.assert_allclose(
            result.ci_lower.data,
            np.percentile(result.samples, 2.5, axis=0),
            rtol=1e-10,
        )

    @pytest.mark.slow
    def test_bootstrap_all_simple_statistics(self, sim_adjacency_multiple):
        """Test all basic statistics work."""
        for statistic in ["mean", "median", "std", "sum", "min", "max"]:
            boot = sim_adjacency_multiple.bootstrap(
                statistic, n_samples=50, random_state=42
            )
            assert isinstance(boot.estimate, Adjacency)

    @pytest.mark.slow
    def test_bootstrap_reproducibility(self, sim_adjacency_multiple):
        """Test same random_state produces identical results."""
        kwargs = {"n_samples": 50, "random_state": 42}

        first = sim_adjacency_multiple.bootstrap("mean", **kwargs)
        second = sim_adjacency_multiple.bootstrap("mean", **kwargs)

        np.testing.assert_allclose(
            first.standard_error.data, second.standard_error.data, rtol=1e-10
        )

    def test_bootstrap_invalid_statistic_error(self, sim_adjacency_multiple):
        """Test error for unsupported statistic."""
        with pytest.raises(ValueError, match="Unsupported statistic"):
            sim_adjacency_multiple.bootstrap("invalid_statistic", n_samples=10)

    @pytest.mark.parametrize("removed", ["stat", "save_boots", "percentiles", "tail"])
    def test_bootstrap_removed_keywords_raise(self, sim_adjacency_multiple, removed):
        with pytest.raises(TypeError):
            sim_adjacency_multiple.bootstrap("mean", n_samples=10, **{removed: 1})

    def test_generate_permutations(self, sim_adjacency_single):
        """Test lazy generation of permuted adjacency matrices."""
        n_permute = 10
        original_data = sim_adjacency_single.data.copy()

        perms = list(
            sim_adjacency_single.generate_permutations(n_permute, random_state=42)
        )
        assert len(perms) == n_permute

        for perm in perms:
            assert isinstance(perm, Adjacency)
            assert perm.shape == sim_adjacency_single.shape

        all_same = all(np.allclose(perm.data, original_data) for perm in perms)
        assert not all_same

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

        perm_a = next(sim_adjacency_single.generate_permutations(1, random_state=1))
        perm_b = next(sim_adjacency_single.generate_permutations(1, random_state=2))
        assert not np.allclose(perm_a.data, perm_b.data)

    def test_regression(self):
        """Test regression with Adjacency and DesignMatrix predictors."""
        m1 = block_diag(np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
        m2 = block_diag(np.zeros((4, 4)), np.ones((4, 4)), np.zeros((4, 4)))
        m3 = block_diag(np.zeros((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
        Y = Adjacency(m1 * 1 + m2 * 2 + m3 * 3, matrix_type="similarity")
        X = Adjacency([m1, m2, m3], matrix_type="similarity")

        stats = Y.regress(X)
        assert np.allclose(stats["beta"], np.array([1, 2, 3]))

        n = 10
        d = Adjacency(
            [
                block_diag(np.ones((4, 4)) + np.eye(4) * 0.1, np.zeros((8, 8)))
                for _ in range(n)
            ],
            matrix_type="similarity",
        )
        X = DesignMatrix(np.ones(n))
        stats = d.regress(X)
        out = stats["beta"].cluster_summary(
            clusters=["Group1"] * 4 + ["Group2"] * 8, scope="within"
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
        assert len(results1["actor_effect"]) == data.n_nodes
        assert results1["relationship_effect"].shape == data.shape
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
            (noisy + noisy.T) / 2,
            matrix_type="similarity",
            labels=["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4,
        )

        clusters = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        cluster_mean = dat.cluster_summary(clusters=clusters)
        for i, j in zip(
            np.array([1, 2, 3]), np.array([cluster_mean[x] for x in cluster_mean])
        ):
            np.testing.assert_almost_equal(i, j, decimal=1)

        for i in dat.cluster_summary(clusters=clusters, scope="between").values():
            np.testing.assert_almost_equal(0, i, decimal=1)

        # v0.6.0: central tendency is `summary=` (canonical vocabulary); the
        # within/between choice is `scope=`. Neither `method=` nor `metric=` exist.
        cluster_median = dat.cluster_summary(clusters=clusters, summary="median")
        for i, j in zip(
            np.array([1, 2, 3]),
            np.array([cluster_median[x] for x in cluster_median]),
        ):
            np.testing.assert_almost_equal(i, j, decimal=1)
        with pytest.raises(TypeError):
            dat.cluster_summary(clusters=clusters, metric="median")
        with pytest.raises(TypeError):
            dat.cluster_summary(clusters=clusters, method="median")
