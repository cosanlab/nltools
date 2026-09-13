"""Tests for nltools.plotting.prediction — the drawing internals behind Roc.plot
and BrainData.predict(plot=True)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nltools.plotting.prediction import (
    plot_class_probability,
    plot_decision_margin,
    plot_predicted_versus_actual,
    plot_roc,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


class TestPlotRoc:
    def test_returns_figure(self):
        """F125: plot_roc returns the matplotlib Figure (was bare return -> None)."""
        fpr = [0.0, 0.1, 0.4, 1.0]
        tpr = [0.0, 0.6, 0.9, 1.0]
        fig = plot_roc(fpr, tpr)
        assert isinstance(fig, plt.Figure)


class TestPlotPredictedVersusActual:
    def test_names_the_correlation_in_the_title(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        axis = plot_predicted_versus_actual(y_true, y_pred, r=0.98)
        assert "r = 0.98" in axis.get_title()


class TestPlotDecisionMargin:
    def test_groups_the_margins_by_class(self):
        margins = np.array([0.4, -0.3, 0.6, -0.5, 0.2, -0.1])
        y_true = np.array([1, 0, 1, 0, 1, 0])
        axis = plot_decision_margin(margins, y_true)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["1", "0"]


class TestPlotClassProbability:
    def test_groups_the_probabilities_by_class(self):
        probabilities = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        y_true = np.array([0, 0, 1, 1, 0, 1])
        axis = plot_class_probability(probabilities, y_true)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["0", "1"]
