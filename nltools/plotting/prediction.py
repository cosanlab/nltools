"""Model output visualization — ROC, SVM margin, regression, and logistic plots."""

__all__ = [
    "plot_dist_from_hyperplane",
    "plot_probability",
    "plot_roc",
    "plot_scatter",
]

import matplotlib.pyplot as plt
import seaborn as sns


def plot_dist_from_hyperplane(stats_output):
    """Plot SVM Classification Distance from Hyperplane.

    Args:
        stats_output (pd.DataFrame): Prediction output table (e.g. from
            `BrainData.predict`).

    Returns:
        seaborn.FacetGrid: Distance from the hyperplane per sample.

    """

    if "dist_from_hyperplane_xval" in stats_output.columns:
        g = sns.catplot(
            data=stats_output,
            x="subject_id",
            y="dist_from_hyperplane_xval",
            hue="Y",
            kind="point",
        )
    else:
        g = sns.catplot(
            data=stats_output,
            x="subject_id",
            y="dist_from_hyperplane_all",
            hue="Y",
            kind="point",
        )
    plt.xlabel("Subject", fontsize=16)
    plt.ylabel("Distance from Hyperplane", fontsize=16)
    plt.title("Classification", fontsize=18)
    return g


def plot_scatter(stats_output):
    """Plot Prediction Scatterplot.

    Args:
        stats_output (pd.DataFrame): Prediction output table (e.g. from
            `BrainData.predict`).

    Returns:
        seaborn.FacetGrid: Scatterplot.

    """

    if "yfit_xval" in stats_output.columns:
        g = sns.lmplot(data=stats_output, x="Y", y="yfit_xval")
    else:
        g = sns.lmplot(data=stats_output, x="Y", y="yfit_all")
    plt.xlabel("Y", fontsize=16)
    plt.ylabel("Predicted Value", fontsize=16)
    plt.title("Prediction", fontsize=18)
    return g


def plot_probability(stats_output):
    """Plot Classification Probability.

    Args:
        stats_output (pd.DataFrame): Prediction output table (e.g. from
            `BrainData.predict`).

    Returns:
        seaborn.FacetGrid: Scatterplot.

    """
    if "Probability_xval" in stats_output.columns:
        g = sns.lmplot(data=stats_output, x="Y", y="Probability_xval", logistic=True)
    else:
        g = sns.lmplot(data=stats_output, x="Y", y="Probability_all", logistic=True)
    plt.xlabel("Y", fontsize=16)
    plt.ylabel("Predicted Probability", fontsize=16)
    plt.title("Prediction", fontsize=18)
    return g


def plot_roc(fpr, tpr):
    """Plot 1-Specificity by Sensitivity.

    Args:
        fpr (np.ndarray): False positive rate per criterion value, from `Roc.calculate`.
        tpr (np.ndarray): True positive rate per criterion value, from `Roc.calculate`.

    Returns:
        matplotlib.figure.Figure: The ROC figure.

    """

    fig = plt.figure()
    plt.plot(fpr, tpr, color="red", linewidth=3)
    plt.xlabel("(1 - Specificity)", fontsize=16)
    plt.ylabel("Sensitivity", fontsize=16)
    plt.title("ROC Plot", fontsize=18)
    return fig
