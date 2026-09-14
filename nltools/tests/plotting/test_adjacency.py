"""Tests for nltools.plotting.adjacency — polars-native label distance and silhouette plots."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from nltools.plotting.adjacency import (
    _plot_between_label_distance,
    _plot_mean_label_distance,
    _plot_silhouette,
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
        out = _plot_mean_label_distance(distance, labels, permutation_test=False)
        assert isinstance(out, pl.DataFrame)
        assert set(out.columns) >= {"Distance", "Group", "Type"}
        # Within values should cluster near 0.1, between near 0.8
        within = out.filter(pl.col("Type") == "Within")["Distance"].to_numpy()
        between = out.filter(pl.col("Type") == "Between")["Distance"].to_numpy()
        assert within.mean() < 0.2
        assert between.mean() > 0.7


class TestPlotBetweenLabelDistance:
    def test_returns_polars_and_within_is_small(self, well_separated_distance):
        distance, labels = well_separated_distance
        long_df, within_mean = _plot_between_label_distance(
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


class TestLongToMatrix:
    def test_integer_labels_reach_the_matrix(self):
        """Polars names pivoted columns by their string rendering, so look them up that way."""
        from nltools.plotting.adjacency import _long_to_matrix

        long_df = pl.DataFrame(
            {
                "label1": [0, 0, 1, 1],
                "label2": [0, 1, 0, 1],
                "mean_distance": [1.0, 4.0, 4.0, 2.0],
            }
        )
        out = _long_to_matrix(
            long_df, "label1", "label2", "mean_distance", np.array([0, 1])
        )
        np.testing.assert_allclose(out, [[1.0, 4.0], [4.0, 2.0]])

    def test_boolean_labels_reach_the_matrix(self):
        """Polars renders `True` as `true`, so the column axis is matched in polars' own spelling."""
        from nltools.plotting.adjacency import _long_to_matrix

        long_df = pl.DataFrame(
            {
                "label1": [False, False, True, True],
                "label2": [False, True, False, True],
                "mean_distance": [1.0, 4.0, 4.0, 2.0],
            }
        )
        out = _long_to_matrix(
            long_df, "label1", "label2", "mean_distance", np.array([False, True])
        )
        np.testing.assert_allclose(out, [[1.0, 4.0], [4.0, 2.0]])

    def test_integer_labels_reach_the_public_heatmap(self):
        """The drawn between-label heatmap carries the real means for integer labels."""
        distance = np.array(
            [
                [0.0, 0.1, 0.8, 0.8],
                [0.1, 0.0, 0.8, 0.8],
                [0.8, 0.8, 0.0, 0.1],
                [0.8, 0.8, 0.1, 0.0],
            ]
        )
        plt.close("all")
        _plot_between_label_distance(
            distance, np.array([0, 0, 1, 1]), permutation_test=False
        )
        drawn = np.asarray(plt.gcf().axes[0].collections[0].get_array())
        assert drawn.max() > 0.5
        plt.close("all")


class TestPlotSilhouette:
    def test_silhouette_scores_positive_for_well_separated(
        self, well_separated_distance
    ):
        distance, labels = well_separated_distance
        out = _plot_silhouette(distance, labels, permutation_test=False)
        assert isinstance(out, pl.DataFrame)
        # Well-separated clusters should have mean silhouette > 0.5
        assert (out["mean_silhouette"] > 0.5).all()

    def test_with_permutation_adds_p_column(self, well_separated_distance):
        distance, labels = well_separated_distance
        out = _plot_silhouette(distance, labels, permutation_test=True, n_permute=200)
        assert isinstance(out, pl.DataFrame)
        assert "p" in out.columns

    def test_draws_on_supplied_ax(self):
        """The silhouette fills land on `ax`, not on whichever axis is current."""
        distance = np.array(
            [
                [0.0, 0.1, 0.8, 0.8],
                [0.1, 0.0, 0.8, 0.8],
                [0.8, 0.8, 0.0, 0.1],
                [0.8, 0.8, 0.1, 0.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])
        _, axs = plt.subplots(1, 2)
        plt.sca(axs[1])
        _plot_silhouette(distance, labels, ax=axs[0], permutation_test=False)
        assert len(axs[0].collections) == 2
        assert len(axs[1].collections) == 0
        plt.close("all")


class TestPlotMDS:
    def test_plot_mds_uses_current_sklearn_api(self, well_separated_distance):
        """`Adjacency.plot_mds` must not trip sklearn>=1.8's MDS deprecations."""
        import warnings

        from nltools.data import Adjacency

        d, labels = well_separated_distance
        adj = Adjacency(d, matrix_type="distance", labels=[str(x) for x in labels])
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            adj.plot_mds(n_components=2, figsize=(4, 4))
        plt.close("all")

    def test_plot_mds_in_three_dimensions(self, well_separated_distance):
        """A 3-D layout must reach three dimensions.

        sklearn's `classical_mds` initializer is built with its own default of
        two components, and `smacof` then adopts the init's width, so asking for
        three and letting sklearn build the init silently yields two. Before the
        fix the 3-D scatter raised `IndexError`; assert the embedding itself, so
        a future fallback to 2-D that fails quietly is caught too.
        """
        from nltools.data import Adjacency

        d, labels = well_separated_distance
        adj = Adjacency(d, matrix_type="distance", labels=[str(x) for x in labels])
        plt.close("all")
        adj.plot_mds(n_components=3, figsize=(4, 4))

        ax = plt.gcf().axes[0]
        # mplot3d keeps the unprojected scatter data only on `_offsets3d`; its
        # public `get_offsets` returns the 2-D screen projection.
        coordinates = ax.collections[0]._offsets3d
        assert len(coordinates) == 3
        for axis_values in coordinates:
            axis_values = np.asarray(axis_values)
            assert axis_values.shape == (adj.n_nodes,)
            assert np.isfinite(axis_values).all()
        plt.close("all")


class TestAdjacencyPlot:
    def test_plot_draws_on_supplied_ax(self, well_separated_distance):
        """`Adjacency.plot(ax=...)` draws the heatmap on the caller's axis."""
        from nltools.data import Adjacency

        distance, _ = well_separated_distance
        adj = Adjacency(distance, matrix_type="distance")
        plt.close("all")
        _, ax = plt.subplots(1)
        n_before = len(plt.get_fignums())
        adj.plot(ax=ax)
        assert len(plt.get_fignums()) == n_before
        assert ax.collections
        plt.close("all")

    def test_plot_anchors_a_signed_matrix_at_zero(self):
        """#527: off-diagonal values that cross zero get a divergent, centered map."""
        from nltools.data import Adjacency

        rng = np.random.default_rng(0)
        matrix = np.corrcoef(rng.normal(size=(6, 20)))
        adj = Adjacency(matrix, matrix_type="similarity")
        off_diagonal = matrix[~np.eye(6, dtype=bool)]
        expected = max(abs(off_diagonal.min()), abs(off_diagonal.max()))

        # Seaborn rebuilds a centered colormap as an unnamed ListedColormap,
        # so compare the ramp by its colors rather than by name.
        positions = np.linspace(0, 1, 5)

        plt.close("all")
        _, ax = plt.subplots(1)
        adj.plot(ax=ax)
        mappable = ax.collections[0]
        assert mappable.get_cmap()(positions) == pytest.approx(
            plt.get_cmap("RdBu_r")(positions)
        )
        assert mappable.get_clim() == pytest.approx((-expected, expected))

        _, ax = plt.subplots(1)
        adj.plot(ax=ax, cmap="viridis", vmin=-1, vmax=1)
        mappable = ax.collections[0]
        assert mappable.get_cmap()(positions) == pytest.approx(
            plt.get_cmap("viridis")(positions)
        )
        assert mappable.get_clim() == pytest.approx((-1.0, 1.0))
        plt.close("all")

    def test_plot_leaves_one_signed_data_sequential(self, well_separated_distance):
        """#527: a matrix that never crosses zero keeps seaborn's sequential default."""
        from nltools.data import Adjacency

        distance, _ = well_separated_distance
        adj = Adjacency(distance, matrix_type="distance")
        square = adj.squareform()

        plt.close("all")
        _, ax = plt.subplots(1)
        adj.plot(ax=ax)
        mappable = ax.collections[0]
        assert mappable.get_cmap().name == "rocket"
        assert mappable.get_clim() == pytest.approx((square.min(), square.max()))
        plt.close("all")

    def test_plot_draws_a_one_panel_stack(self):
        """A one-row stack, and `limit=1` on any stack, still draws its panel."""
        from nltools.data import Adjacency

        adj = Adjacency(np.array([1.0, 2.0, 3.0]))
        stack = adj.append(adj)

        plt.close("all")
        adj[[0]].plot()
        assert len(plt.gcf().axes) == 2  # one heatmap plus its colorbar
        plt.close("all")
        stack.plot(limit=1)
        assert len(plt.gcf().axes) == 2
        plt.close("all")
        adj[[0]].similarity(adj, method=None, plot=True, n_jobs=1)
        assert plt.gcf().axes
        plt.close("all")

    def test_plot_gives_every_panel_of_a_shared_label_stack_all_the_labels(self):
        """Shared labels describe nodes, so each panel of a stack carries all of them."""
        from nltools.data import Adjacency

        labels = ["a", "b", "c"]
        matrix = Adjacency(np.array([1.0, 2.0, 3.0]), labels=labels)
        stack = matrix.append(Adjacency(np.array([4.0, 5.0, 6.0]), labels=labels))

        plt.close("all")
        stack.plot(cbar=False)
        for ax in plt.gcf().axes:
            assert [t.get_text() for t in ax.get_xticklabels()] == labels
            assert [t.get_text() for t in ax.get_yticklabels()] == labels
        plt.close("all")

        nested = Adjacency([matrix, matrix], labels=[["a", "b", "c"], ["d", "e", "f"]])
        nested.plot(cbar=False)
        panels = plt.gcf().axes
        assert [t.get_text() for t in panels[0].get_xticklabels()] == ["a", "b", "c"]
        assert [t.get_text() for t in panels[1].get_xticklabels()] == ["d", "e", "f"]
        plt.close("all")


def _correlation_adjacency(seed, n_nodes=6):
    """A signed subject x subject similarity matrix, the ISRSA case."""
    from nltools.data import Adjacency

    rng = np.random.default_rng(seed)
    return Adjacency(
        np.corrcoef(rng.normal(size=(n_nodes, 20))), matrix_type="similarity"
    )


class TestAdjacencyPlotStacked:
    def test_each_matrix_fills_its_own_triangle(self):
        """#529: self is the upper triangle, `data` the lower, diagonal masked in both."""
        upper_adj = _correlation_adjacency(0)
        lower_adj = _correlation_adjacency(1)
        n = upper_adj.n_nodes

        plt.close("all")
        ax = upper_adj.plot_stacked(lower_adj)
        assert len(ax.collections) == 2
        upper_mesh, lower_mesh = ax.collections

        ones = np.ones((n, n), dtype=bool)
        assert np.array_equal(np.ma.getmaskarray(upper_mesh.get_array()), np.tril(ones))
        assert np.array_equal(np.ma.getmaskarray(lower_mesh.get_array()), np.triu(ones))

        iu = np.triu_indices(n, k=1)
        il = np.tril_indices(n, k=-1)
        assert np.asarray(upper_mesh.get_array())[iu] == pytest.approx(
            upper_adj.squareform()[iu]
        )
        assert np.asarray(lower_mesh.get_array())[il] == pytest.approx(
            lower_adj.squareform()[il]
        )
        plt.close("all")

    def test_titles_sit_above_and_below_the_square(self):
        """#529: the upper matrix is titled above the panel, the lower one below."""
        upper_adj = _correlation_adjacency(0)
        lower_adj = _correlation_adjacency(1)

        plt.close("all")
        ax = upper_adj.plot_stacked(
            lower_adj, upper_title="Neural similarity", lower_title="Rated similarity"
        )
        assert ax.get_title() == "Neural similarity"
        assert ax.get_xlabel() == "Rated similarity"
        plt.close("all")

    def test_one_colorbar_when_both_triangles_share_a_scale(self):
        """#529: a shared colormap and limits need only one bar."""
        upper_adj = _correlation_adjacency(0)
        lower_adj = _correlation_adjacency(1)

        plt.close("all")
        ax = upper_adj.plot_stacked(lower_adj, cmap="RdBu_r", vmin=-1, vmax=1)
        assert len(ax.child_axes) == 1
        plt.close("all")

    def test_two_colorbars_when_the_colormaps_differ(self):
        """#529: triangles on different ramps each carry their own bar."""
        upper_adj = _correlation_adjacency(0)
        lower_adj = _correlation_adjacency(1)

        plt.close("all")
        ax = upper_adj.plot_stacked(
            lower_adj, cmap=("RdBu_r", "viridis"), vmin=-1, vmax=1
        )
        assert len(ax.child_axes) == 2
        plt.close("all")

    def test_mismatched_node_counts_raise(self):
        """#529: the two matrices must describe the same nodes."""
        upper_adj = _correlation_adjacency(0, n_nodes=6)
        lower_adj = _correlation_adjacency(1, n_nodes=5)

        with pytest.raises(ValueError, match="same nodes"):
            upper_adj.plot_stacked(lower_adj)

    def test_similarity_plot_draws_the_stacked_figure(self):
        """#529: `similarity(plot=True)` reaches the same two-triangle figure."""
        upper_adj = _correlation_adjacency(0)
        lower_adj = _correlation_adjacency(1)

        plt.close("all")
        upper_adj.similarity(lower_adj, n_permute=0, plot=True)
        ax = plt.gcf().axes[0]
        assert len(ax.collections) == 2
        plt.close("all")
