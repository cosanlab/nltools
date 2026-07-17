"""Tests for nltools.plotting.adjacency — polars-native label distance and silhouette plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from nltools.plotting.adjacency import (
    _stacked_adjacency_matrix,
    plot_between_label_distance,
    plot_mean_label_distance,
    plot_silhouette,
    plot_stacked_adjacency,
)


@pytest.fixture
def well_separated_distance():
    """3 clusters of 3 points each: within-cluster ~0.1, between ~0.8."""
    rng = np.random.default_rng(42)
    n = 9
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if labels[i] == labels[j]:
                d[i, j] = 0.1 + rng.random() * 0.05
            else:
                d[i, j] = 0.8 + rng.random() * 0.05
    d = (d + d.T) / 2
    return d, labels


class TestPlotMeanLabelDistance:
    def test_returns_polars_long_format(self, well_separated_distance):
        distance, labels = well_separated_distance
        out = plot_mean_label_distance(distance, labels, permutation_test=False)
        assert isinstance(out, pl.DataFrame)
        assert set(out.columns) >= {"Distance", "Group", "Type"}
        # Within values should cluster near 0.1, between near 0.8
        within = out.filter(pl.col("Type") == "Within")["Distance"].to_numpy()
        between = out.filter(pl.col("Type") == "Between")["Distance"].to_numpy()
        assert within.mean() < 0.2
        assert between.mean() > 0.7

    def test_with_permutation_returns_stats(self, well_separated_distance):
        distance, labels = well_separated_distance
        out, stats = plot_mean_label_distance(
            distance, labels, permutation_test=True, n_permute=200
        )
        assert isinstance(out, pl.DataFrame)
        assert isinstance(stats, dict)
        assert set(stats.keys()) == {"0", "1", "2"}


class TestPlotBetweenLabelDistance:
    def test_returns_polars_and_within_is_small(self, well_separated_distance):
        distance, labels = well_separated_distance
        long_df, within_mean = plot_between_label_distance(
            distance, labels, permutation_test=False
        )
        assert isinstance(long_df, pl.DataFrame)
        assert isinstance(within_mean, pl.DataFrame)
        # within_mean rows are labels, columns are labels; diagonal = within-cluster mean
        # Use long form: label1 == label2 entries should be small
        diag = within_mean.filter(pl.col("label1") == pl.col("label2"))[
            "mean_distance"
        ].to_numpy()
        off = within_mean.filter(pl.col("label1") != pl.col("label2"))[
            "mean_distance"
        ].to_numpy()
        assert diag.mean() < 0.2
        assert off.mean() > 0.7


class TestPlotSilhouette:
    def test_silhouette_scores_positive_for_well_separated(
        self, well_separated_distance
    ):
        distance, labels = well_separated_distance
        out = plot_silhouette(distance, labels, permutation_test=False)
        assert isinstance(out, pl.DataFrame)
        # Well-separated clusters should have mean silhouette > 0.5
        assert (out["mean_silhouette"] > 0.5).all()

    def test_with_permutation_adds_p_column(self, well_separated_distance):
        distance, labels = well_separated_distance
        out = plot_silhouette(distance, labels, permutation_test=True, n_permute=200)
        assert isinstance(out, pl.DataFrame)
        assert "p" in out.columns


class TestPlotStackedAdjacency:
    def test_runs(self):
        from nltools.data import Adjacency

        rng = np.random.default_rng(0)
        a1 = Adjacency(rng.random((6, 6)), matrix_type="similarity")
        a2 = Adjacency(rng.random((6, 6)), matrix_type="similarity")
        ax = plot_stacked_adjacency(a1, a2)
        assert ax is not None

    def test_consistent_triangle_mapping_across_normalize(self):
        """F126: adjacency1 must drive the same triangle regardless of normalize.

        Build two Adjacency inputs whose off-diagonal orderings are opposites, so
        the identity of whichever input lands in the upper triangle is recoverable
        from the value ordering. That ordering must match adjacency1 in BOTH the
        normalize=False and normalize=True branches (mean-centering + positive
        scaling is monotonic, so argsort is preserved).
        """
        from nltools.data import Adjacency

        n = 4
        # Upper-triangle (squareform) vectors that are strict reverses of each other.
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v2 = v1[::-1].copy()
        a1 = Adjacency(v1, matrix_type="similarity_flat")
        a2 = Adjacency(v2, matrix_type="similarity_flat")

        iu = np.triu_indices(n, k=1)
        a1_order = np.argsort(a1.squareform()[iu])

        raw = _stacked_adjacency_matrix(a1, a2, normalize=False)
        norm = _stacked_adjacency_matrix(a1, a2, normalize=True)

        # Upper triangle tracks adjacency1 in both branches (same argsort).
        assert np.array_equal(np.argsort(raw[iu]), a1_order)
        assert np.array_equal(np.argsort(norm[iu]), a1_order)

    def test_normalize_no_nan_when_triangle_all_negative(self):
        """F126: normalizing must divide by max-abs, never producing inf/nan."""
        from nltools.data import Adjacency

        rng = np.random.default_rng(3)
        a1 = Adjacency(rng.random((5, 5)), matrix_type="similarity")
        a2 = Adjacency(rng.random((5, 5)), matrix_type="similarity")
        out = _stacked_adjacency_matrix(a1, a2, normalize=True)
        assert np.isfinite(out).all()


class TestPlotBetweenLabelDistanceFigureLeak:
    def test_no_stray_figure_when_ax_supplied(self, well_separated_distance):
        """F127: supplying an ax must not spawn an extra blank figure."""
        distance, labels = well_separated_distance
        plt.close("all")
        fig, ax = plt.subplots(1)
        n_before = len(plt.get_fignums())
        plot_between_label_distance(distance, labels, ax=ax, permutation_test=False)
        assert len(plt.get_fignums()) == n_before
        plt.close("all")
