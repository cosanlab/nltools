"""Visualize a DesignMatrix as a heatmap, overlaid time courses, or a correlation matrix.

`DesignMatrix.plot` dispatches over `method` to `plot_matrix`,
`plot_timeseries`, and `plot_corr`, mirroring `BrainData.plot`.
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
    """Visualize a DesignMatrix, dispatching over `method`.

    Args:
        dm (DesignMatrix): DesignMatrix instance.
        method (str): ``'matrix'`` (SPM-style heatmap), ``'timeseries'``
            (overlaid line plot), or ``'corr'`` (correlation heatmap).
            Default: ``'matrix'``.
        columns (list[str] | None): Subset of columns to plot. Defaults to all.
        rescale (bool): ``'matrix'`` only; rescale each column by its L2 norm.
            Default: True.
        metric (str): ``'corr'`` only; ``'pearson'`` (default) or ``'spearman'``.
        ax (matplotlib.axes.Axes | None): Existing axis to draw on; a new
            figure is created if omitted.
        figsize (tuple | None): Figure size; per-method default when omitted.
        title (str | None): Axis title.
        cmap (str | None): Colormap (``'matrix'`` / ``'corr'``).
        save (str | None): Path to save the figure.
        **kwargs (dict): Forwarded to the underlying plotter
            (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``;
            ``matplotlib.axes.Axes.plot`` for ``'timeseries'``).

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.

    Raises:
        ValueError: If `method` is not one of the three supported values.
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
        dm (DesignMatrix): DesignMatrix instance.
        columns (list[str] | None): Subset of columns to plot. Defaults to all columns.
        rescale (bool): If True, rescale each column by its L2 norm so columns
            with different native magnitudes are visually comparable
            (SPM/nilearn convention). Default: True.
        figsize (tuple | None): Figure size; defaults to ``(4, 6)`` when a new
            figure is made.
        title (str | None): Axis title.
        cmap (str | None): Colormap name. Default: ``'gray'``.
        ax (matplotlib.axes.Axes | None): Existing axis to draw on; a new
            figure is created if omitted.
        save (str | None): Path to save the figure.
        **kwargs (dict): Forwarded to ``seaborn.heatmap``.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
    """
    import seaborn as sns

    labels = dm.columns if columns is None else list(columns)
    values = dm.data.select(labels).to_numpy()
    if rescale:
        values = values.astype(float)
        values = values / np.maximum(1.0e-12, np.sqrt(np.sum(values**2, 0)))

    fig, ax, owns_fig = _new_axis(ax, figsize or (4, 6))
    heatmap_kwargs = {
        "cmap": cmap or "gray",
        "cbar": False,
        "xticklabels": labels,
        "yticklabels": False,  # Too many rows for labels typically
    }
    heatmap_kwargs.update(kwargs)
    sns.heatmap(values, ax=ax, **heatmap_kwargs)

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
        dm (DesignMatrix): DesignMatrix instance.
        columns (list[str] | None): Subset of columns to plot. Defaults to all columns.
        figsize (tuple | None): Figure size; defaults to ``(8, 4)`` when a new
            figure is made.
        title (str | None): Axis title.
        ax (matplotlib.axes.Axes | None): Existing axis to draw on; a new
            figure is created if omitted.
        save (str | None): Path to save the figure.
        **kwargs (dict): Forwarded to ``matplotlib.axes.Axes.plot`` for each line.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
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
        dm (DesignMatrix): DesignMatrix instance.
        columns (list[str] | None): Subset of columns to correlate. Defaults to
            all columns.
        metric (str): ``'pearson'`` (default) or ``'spearman'``.
        figsize (tuple | None): Figure size; scales with the number of columns
            when omitted.
        title (str | None): Axis title.
        cmap (str | None): Colormap name. Default: ``'RdBu_r'``.
        ax (matplotlib.axes.Axes | None): Existing axis to draw on; a new
            figure is created if omitted.
        save (str | None): Path to save the figure.
        **kwargs (dict): Forwarded to ``seaborn.heatmap`` (e.g. ``annot=False``).

    Returns:
        matplotlib.figure.Figure: The rendered figure.
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
