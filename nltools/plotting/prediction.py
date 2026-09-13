"""Model output visualization — ROC curves and the cross-validated decoding figures.

Every function here takes arrays, not a result table: `BrainData.predict` returns a
`Predict` record plus the row-aligned out-of-fold decision values that go with it,
and these draw from those directly.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_predicted_versus_actual(y_true, y_pred, *, r=None, ax=None):
    """Scatter cross-validated predictions against the observed target.

    Args:
        y_true (array-like): Observed target, one value per observation.
        y_pred (array-like): Out-of-fold prediction for the same observations.
        r (float, optional): Cross-validated Pearson correlation to name in the
            title; omitted from the title when `None`.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; a new figure is
            created when omitted.

    Returns:
        matplotlib.axes.Axes: The axis holding the scatter and its fit line.
    """
    if ax is None:
        _, ax = plt.subplots(1)
    axis = sns.regplot(
        x=np.asarray(y_true, dtype=float), y=np.asarray(y_pred, dtype=float), ax=ax
    )
    axis.set_xlabel("Observed", fontsize=16)
    axis.set_ylabel("Predicted", fontsize=16)
    title = "Predicted vs actual" if r is None else f"Predicted vs actual (r = {r:.2f})"
    axis.set_title(title, fontsize=18)
    return axis


def plot_decision_margin(margins, y_true, *, ax=None):
    """Plot each observation's signed distance from the decision boundary by class.

    Args:
        margins (array-like): Out-of-fold `decision_function` values, one per
            observation.
        y_true (array-like): Class label for the same observations.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; a new figure is
            created when omitted.

    Returns:
        matplotlib.axes.Axes: The axis holding the margin figure.
    """
    if ax is None:
        _, ax = plt.subplots(1)
    axis = sns.stripplot(
        x=np.asarray(y_true).astype(str),
        y=np.asarray(margins, dtype=float),
        ax=ax,
    )
    axis.axhline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Class", fontsize=16)
    axis.set_ylabel("Distance from Hyperplane", fontsize=16)
    axis.set_title("Classification margin", fontsize=18)
    return axis


def plot_class_probability(probabilities, y_true, *, ax=None):
    """Plot the predicted positive-class probability of each observation by class.

    Args:
        probabilities (array-like): Out-of-fold `predict_proba` values for the
            positive class, one per observation.
        y_true (array-like): Class label for the same observations.
        ax (matplotlib.axes.Axes, optional): Axis to draw on; a new figure is
            created when omitted.

    Returns:
        matplotlib.axes.Axes: The axis holding the probability figure.
    """
    if ax is None:
        _, ax = plt.subplots(1)
    axis = sns.stripplot(
        x=np.asarray(y_true).astype(str),
        y=np.asarray(probabilities, dtype=float),
        ax=ax,
    )
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlabel("Class", fontsize=16)
    axis.set_ylabel("Predicted Probability", fontsize=16)
    axis.set_title("Classification probability", fontsize=18)
    return axis


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
