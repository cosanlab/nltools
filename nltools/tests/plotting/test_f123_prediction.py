"""Regression tests for nltools.plotting.prediction (findings F123-F125).

These cover the model-output plotters that crashed under seaborn 0.13.2 (positional
x/y arguments) and the return-None-vs-docstring mismatches.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from seaborn.axisgrid import FacetGrid

from nltools.plotting.prediction import (
    plot_dist_from_hyperplane,
    plot_probability,
    plot_roc,
    plot_scatter,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


@pytest.fixture
def hyperplane_stats():
    """Minimal SVM-margin output frame shaped like predict() produces."""
    return pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2, 3, 3],
            "dist_from_hyperplane_xval": [0.4, -0.3, 0.6, -0.5, 0.2, -0.1],
            "Y": [1, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def scatter_stats():
    return pd.DataFrame(
        {
            "Y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "yfit_xval": [1.1, 1.9, 3.2, 3.8, 5.1],
        }
    )


@pytest.fixture
def probability_stats():
    return pd.DataFrame(
        {
            "Y": [0, 0, 1, 1, 0, 1, 1, 0],
            "Probability_xval": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15],
        }
    )


class TestPlotDistFromHyperplane:
    def test_returns_facetgrid_no_positional_crash(self, hyperplane_stats):
        """F123: keyword x/y; returns the seaborn FacetGrid (was TypeError + None)."""
        g = plot_dist_from_hyperplane(hyperplane_stats)
        assert isinstance(g, FacetGrid)

    def test_all_branch(self, hyperplane_stats):
        stats = hyperplane_stats.rename(
            columns={"dist_from_hyperplane_xval": "dist_from_hyperplane_all"}
        )
        g = plot_dist_from_hyperplane(stats)
        assert isinstance(g, FacetGrid)


class TestPlotScatter:
    def test_returns_facetgrid(self, scatter_stats):
        """F125: plot_scatter returns its FacetGrid instead of None."""
        g = plot_scatter(scatter_stats)
        assert isinstance(g, FacetGrid)


class TestPlotProbability:
    def test_no_data_binding_crash(self, probability_stats):
        """F124: keyword x/y. logistic=True needs statsmodels (optional).

        Before the fix this raised ``TypeError: lmplot() got multiple values for
        argument 'data'`` during argument binding. After the fix that binding
        succeeds: with statsmodels installed we get a FacetGrid; without it,
        seaborn raises a RuntimeError naming statsmodels -- either way we proved
        the positional-argument bug is gone.
        """
        try:
            g = plot_probability(probability_stats)
        except RuntimeError as e:
            assert "statsmodels" in str(e)
        else:
            assert isinstance(g, FacetGrid)


class TestPlotRoc:
    def test_returns_figure(self):
        """F125: plot_roc returns the matplotlib Figure (was bare return -> None)."""
        fpr = [0.0, 0.1, 0.4, 1.0]
        tpr = [0.0, 0.6, 0.9, 1.0]
        fig = plot_roc(fpr, tpr)
        assert isinstance(fig, plt.Figure)
