"""DesignMatrix visualization functions.

Standalone functions extracted from ``DesignMatrix`` methods. Each takes a
``DesignMatrix`` instance (``dm``) as its first argument. ``DesignMatrix.plot``
dispatches over ``method`` to the helpers here, mirroring ``BrainData.plot``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from nltools.data.designmatrix import DesignMatrix


VALID_PLOT_METHODS = ("matrix", "timeseries", "corr")


def plot_designmatrix(
    dm: DesignMatrix,
    method: str = "matrix",
    *,
    columns: list[str] | None = None,
    rescale: bool = True,
    metric: str = "pearson",
    ax: plt.Axes | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    cmap: str | None = None,
    save: str | None = None,
    **kwargs,
):
    """Visualize a DesignMatrix, dispatching over ``method``.

    See `DesignMatrix.plot` for the full argument documentation.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    if method == "matrix":
        return plot_matrix(
            dm,
            columns=columns,
            rescale=rescale,
            figsize=figsize,
            title=title,
            cmap=cmap,
            ax=ax,
            save=save,
            **kwargs,
        )
    if method == "timeseries":
        return plot_timeseries(
            dm,
            columns=columns,
            figsize=figsize,
            title=title,
            ax=ax,
            save=save,
            **kwargs,
        )
    if method == "corr":
        return plot_corr(
            dm,
            columns=columns,
            metric=metric,
            figsize=figsize,
            title=title,
            cmap=cmap,
            ax=ax,
            save=save,
            **kwargs,
        )
    raise ValueError(f"Invalid method {method!r}. Must be one of {VALID_PLOT_METHODS}.")


def plot_matrix(
    dm: DesignMatrix,
    *,
    columns: list[str] | None = None,
    rescale: bool = True,
    figsize: tuple | None = None,
    title: str | None = None,
    cmap: str | None = None,
    ax: plt.Axes | None = None,
    save: str | None = None,
    **kwargs,
):
    """Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).

    Args:
        dm: DesignMatrix instance.
        columns: Subset of columns to plot. Defaults to all columns.
        rescale: If True, rescale each column by its L2 norm so columns with
            different native magnitudes are visually comparable (SPM/nilearn
            convention). Default: True.
        figsize: Figure size; defaults to ``(4, 6)`` when a new figure is made.
        title: Optional axis title.
        cmap: Colormap name. Default: ``'gray'``.
        ax: Existing axis to draw on; a new figure is created if omitted.
        save: Optional path to save the figure.
        **kwargs: Forwarded to ``seaborn.heatmap``.

    Returns:
        matplotlib.figure.Figure
    """
    import pandas as pd
    import seaborn as sns

    from .io import to_pandas

    df = to_pandas(dm)
    if columns is not None:
        df = df[list(columns)]
    if rescale:
        X = df.to_numpy(dtype=float)
        X = X / np.maximum(1.0e-12, np.sqrt(np.sum(X**2, 0)))
        df = pd.DataFrame(X, columns=df.columns)

    fig, ax, owns_fig = _new_axis(ax, figsize or (4, 6))
    heatmap_kwargs = {
        "cmap": cmap or "gray",
        "cbar": False,
        "yticklabels": False,  # Too many rows for labels typically
    }
    heatmap_kwargs.update(kwargs)
    sns.heatmap(df, ax=ax, **heatmap_kwargs)

    ax.set_xlabel("Regressors")
    ax.set_ylabel("Time (TRs)")
    if title:
        ax.set_title(title)
    return _finalize(fig, owns_fig, save)


def plot_timeseries(
    dm: DesignMatrix,
    *,
    columns: list[str] | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    save: str | None = None,
    **kwargs,
):
    """Plot regressor time courses as overlaid lines.

    One line is drawn per column. Pass the same ``ax`` across calls to overlay
    multiple DesignMatrices (e.g. original vs. convolved).

    Args:
        dm: DesignMatrix instance.
        columns: Subset of columns to plot. Defaults to all columns.
        figsize: Figure size; defaults to ``(8, 4)`` when a new figure is made.
        title: Optional axis title.
        ax: Existing axis to draw on; a new figure is created if omitted.
        save: Optional path to save the figure.
        **kwargs: Forwarded to ``matplotlib.axes.Axes.plot`` for each line.

    Returns:
        matplotlib.figure.Figure
    """
    cols = list(columns) if columns is not None else list(dm.columns)

    fig, ax, owns_fig = _new_axis(ax, figsize or (8, 4))
    x = np.arange(dm.shape[0])
    for col in cols:
        ax.plot(x, dm.data[col].to_numpy(), label=col, **kwargs)

    ax.set_xlabel("Time (TRs)")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    return _finalize(fig, owns_fig, save)


def plot_corr(
    dm: DesignMatrix,
    *,
    columns: list[str] | None = None,
    metric: str = "pearson",
    figsize: tuple | None = None,
    title: str | None = None,
    cmap: str | None = None,
    ax: plt.Axes | None = None,
    save: str | None = None,
    **kwargs,
):
    """Render a labeled correlation heatmap of the columns.

    Reuses `DesignMatrix.corr`, which returns a similarity ``Adjacency``
    with the unit diagonal dropped; the diagonal is restored to ``1.0`` here so
    the heatmap reads as a standard correlation matrix.

    Args:
        dm: DesignMatrix instance.
        columns: Subset of columns to correlate. Defaults to all columns.
        metric: ``'pearson'`` (default) or ``'spearman'``.
        figsize: Figure size; scales with the number of columns when omitted.
        title: Optional axis title.
        cmap: Colormap name. Default: ``'RdBu_r'``.
        ax: Existing axis to draw on; a new figure is created if omitted.
        save: Optional path to save the figure.
        **kwargs: Forwarded to ``seaborn.heatmap`` (e.g. ``annot=False``).

    Returns:
        matplotlib.figure.Figure
    """
    import seaborn as sns

    from .diagnostics import corr as _corr

    adj = _corr(dm, metric=metric, columns=columns)
    mat = adj.squareform()
    np.fill_diagonal(mat, 1.0)  # restore unit diagonal dropped by Adjacency
    labels = list(adj.labels) if adj.labels else "auto"

    n = mat.shape[0]
    side = max(4.0, 0.6 * n + 2.0)
    fig, ax, owns_fig = _new_axis(ax, figsize or (side, side))
    heatmap_kwargs = {
        "cmap": cmap or "RdBu_r",
        "vmin": -1.0,
        "vmax": 1.0,
        "square": True,
        "annot": True,
        "fmt": ".2f",
        "xticklabels": labels,
        "yticklabels": labels,
    }
    heatmap_kwargs.update(kwargs)
    sns.heatmap(mat, ax=ax, **heatmap_kwargs)
    if title:
        ax.set_title(title)
    return _finalize(fig, owns_fig, save)


def _new_axis(ax, figsize):
    """Resolve a drawing axis, tracking whether we created its figure.

    Caller-supplied axes belong to the caller's figure lifecycle, so we don't
    detach/close them in ``_finalize``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _finalize(fig, owns_fig, save):
    """Save if requested and detach owned figures from pyplot.

    Detaching keeps the notebook ``flush_figures`` post-hook from rendering the
    returned figure a second time alongside its ``_repr_*_`` display.
    """
    import matplotlib.pyplot as plt

    if save:
        fig.savefig(save, bbox_inches="tight")
    if owns_fig:
        plt.close(fig)
    return fig
