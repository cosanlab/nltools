"""Plotting functions for Adjacency matrices."""

import numpy as np


def _divergent_heatmap_defaults(square):
    """Heatmap keywords that anchor a signed matrix at zero, or none if one-signed.

    Seaborn picks a sequential ramp and data-range limits whenever `center` is
    unset, which leaves a matrix straddling zero with no visible anchor there.
    The diagonal is excluded because it is a stored constant (0 for a distance,
    1 for a similarity), not data, and it would otherwise set the limits.

    Args:
        square (np.ndarray): Square matrix about to be drawn.

    Returns:
        dict: `cmap`, `center`, `vmin` and `vmax` for a signed matrix; empty
            for a one-signed, empty or all-NaN one.
    """
    off_diagonal = square[~np.eye(square.shape[0], dtype=bool)]
    if off_diagonal.size == 0 or np.all(np.isnan(off_diagonal)):
        return {}
    low, high = np.nanmin(off_diagonal), np.nanmax(off_diagonal)
    if not (low < 0 < high):
        return {}
    limit = float(max(abs(low), abs(high)))
    return {"cmap": "RdBu_r", "center": 0, "vmin": -limit, "vmax": limit}


def _heatmap_kwargs(square, kwargs):
    """Merge the divergent defaults under the caller's own heatmap keywords."""
    merged = dict(kwargs)
    for name, value in _divergent_heatmap_defaults(square).items():
        merged.setdefault(name, value)
    return merged


def _plot_adjacency(adj, *, limit=3, ax=None, **kwargs):
    """Create a heatmap of an Adjacency matrix.

    Signed matrices are anchored at zero (see `_divergent_heatmap_defaults`);
    caller keywords always win.

    Args:
        adj (Adjacency): Adjacency object to plot.
        limit (int): Number of heatmaps to plot if the object contains multiple
            matrices. Default 3.
        ax (matplotlib.axes.Axes, optional): Axis to draw on (single matrix only).
        **kwargs (dict): Forwarded to `seaborn.heatmap`.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if adj.is_single_matrix:
        if ax is None:
            _, ax = plt.subplots(nrows=1, figsize=(7, 5))
        square = adj.squareform()
        heatmap_kwargs = _heatmap_kwargs(square, kwargs)
        if adj.labels:
            sns.heatmap(
                square,
                square=True,
                ax=ax,
                xticklabels=adj.labels,
                yticklabels=adj.labels,
                **heatmap_kwargs,
            )
        else:
            sns.heatmap(square, square=True, ax=ax, **heatmap_kwargs)
    else:
        if ax is not None:
            print("ax is ignored when plotting multiple images")
        n_subs = np.minimum(len(adj), limit)
        _, a = plt.subplots(nrows=n_subs, figsize=(7, len(adj) * 5))
        for i in range(n_subs):
            square = adj[i].squareform()
            heatmap_kwargs = _heatmap_kwargs(square, kwargs)
            if adj.labels:
                sns.heatmap(
                    square,
                    square=True,
                    xticklabels=adj.labels[i],
                    yticklabels=adj.labels[i],
                    ax=a[i],
                    **heatmap_kwargs,
                )
            else:
                sns.heatmap(square, square=True, ax=a[i], **heatmap_kwargs)
    return


def _plot_mds(
    adj,
    *,
    n_components=2,
    metric_mds=True,
    labels=None,
    labels_color=None,
    cmap=None,
    view=(30, 20),
    figsize=None,
    ax=None,
    n_jobs=-1,
    **kwargs,
):
    """Plot multidimensional scaling.

    Args:
        adj (Adjacency): Adjacency object to plot (must be a single distance matrix).
        n_components (int): Number of dimensions to project (2 or 3).
        metric_mds (bool): Perform metric (True) or non-metric (False) scaling.
            Default True.
        labels (list, optional): Overrides the labels stored on `adj`.
        labels_color (list, optional): One color per label.
        cmap (matplotlib.colors.Colormap, optional): Colormap. Default `plt.cm.hot_r`.
        view (tuple): Elevation/azimuth for a 3-D plot. Default (30, 20).
        figsize (list): Figure size. Default [12, 8].
        ax (matplotlib.axes.Axes, optional): Axis to draw on.
        n_jobs (int): Number of parallel jobs.
        **kwargs (dict): Forwarded to `sklearn.manifold.MDS`.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS, ClassicalMDS

    if cmap is None:
        cmap = plt.cm.hot_r
    if figsize is None:
        figsize = [12, 8]

    if adj.matrix_type != "distance":
        raise ValueError("MDS only works on distance matrices.")
    if not adj.is_single_matrix:
        raise ValueError("MDS only works on single matrices.")
    if n_components not in [2, 3]:
        raise ValueError(f"Cannot plot {n_components}-d image")
    if labels is not None:
        if len(labels) != adj.n_nodes:
            raise ValueError(
                "Make sure labels matches the same shape as Adjacency data"
            )
    else:
        labels = adj.labels
    if labels_color is not None:
        if len(labels) == 0:
            raise ValueError("Make sure that Adjacency object has labels specified.")
        if len(labels) != len(labels_color):
            raise ValueError("Length of labels_color must match self.labels.")

    # Run MDS (sklearn >= 1.8 API). The classical-MDS starting configuration is
    # built here, at the requested width, and passed to `fit_transform`, which
    # takes precedence over the constructor's `init` — sklearn skips building
    # its own, so this is computed once. Asking the constructor for it instead
    # gives a 2-D start whatever `n_components` says, because it builds its
    # `ClassicalMDS` with that class's own default, and `smacof` then adopts the
    # start's width: a 3-D request would silently come back 2-D. `init` and
    # `n_init` are still named because omitting either warns until they become
    # sklearn's defaults in 1.9/1.10; classical MDS is deterministic, so one run
    # suffices.
    square = adj.squareform()
    init = ClassicalMDS(n_components=n_components, metric="precomputed").fit_transform(
        square
    )
    mds = MDS(
        n_components=n_components,
        metric_mds=metric_mds,
        n_jobs=n_jobs,
        metric="precomputed",
        init="classical_mds",
        n_init=1,
        **kwargs,
    )
    proj = mds.fit_transform(square, init=init)

    # Create Plot
    if ax is None:  # Create axis
        fig = plt.figure(figsize=figsize)
        if n_components == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(*view)
        elif n_components == 2:
            ax = fig.add_subplot(111)

    # Plot dots
    if n_components == 3:
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], s=1, c="k")
    elif n_components == 2:
        ax.scatter(proj[:, 0], proj[:, 1], s=1, c="k")

    # Plot labels
    if labels_color is None:
        labels_color = ["black"] * len(labels)
    if n_components == 3:
        for (x, y, z), label, color in zip(proj, labels, labels_color):
            ax.text(
                x,
                y,
                z,
                label,
                color="white",
                bbox={"facecolor": color, "alpha": 1, "boxstyle": "round,pad=0.3"},
            )
    else:
        for (x, y), label, color in zip(proj, labels, labels_color):
            ax.text(
                x,
                y,
                label,
                color="white",  # color,
                bbox={"facecolor": color, "alpha": 1, "boxstyle": "round,pad=0.3"},
            )

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
