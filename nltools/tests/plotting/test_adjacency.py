"""Tests for nltools.plotting.adjacency — polars-native label distance and silhouette plots."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

from nltools.plotting.adjacency import (
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
