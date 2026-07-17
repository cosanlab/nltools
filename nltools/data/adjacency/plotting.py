"""Plotting functions for Adjacency matrices."""

import numpy as np


def plot_adjacency(adj, limit=3, axes=None, *args, **kwargs):
    """Create Heatmap of Adjacency Matrix.

    Can pass in any ``sns.heatmap`` argument.

    Args:
        adj (Adjacency): Adjacency object to plot.
        limit (int): Number of heatmaps to plot if object contains multiple adjacencies (default: 3).
        axes: Matplotlib axis handle.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if adj.is_single_matrix:
        if axes is None:
            _, axes = plt.subplots(nrows=1, figsize=(7, 5))
        if adj.labels:
            sns.heatmap(
                adj.squareform(),
                square=True,
                ax=axes,
                xticklabels=adj.labels,
                yticklabels=adj.labels,
                *args,
                **kwargs,
            )
        else:
            sns.heatmap(adj.squareform(), square=True, ax=axes, *args, **kwargs)
    else:
        if axes is not None:
            print("axes is ignored when plotting multiple images")
        n_subs = np.minimum(len(adj), limit)
        _, a = plt.subplots(nrows=n_subs, figsize=(7, len(adj) * 5))
        for i in range(n_subs):
            if adj.labels:
                sns.heatmap(
                    adj[i].squareform(),
                    square=True,
                    xticklabels=adj.labels[i],
                    yticklabels=adj.labels[i],
                    ax=a[i],
                    *args,
                    **kwargs,
                )
            else:
                sns.heatmap(adj[i].squareform(), square=True, ax=a[i], *args, **kwargs)
    return


def plot_mds(
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
    """Plot Multidimensional Scaling.

    Args:
        adj (Adjacency): Adjacency object to plot (must be a distance matrix).
        n_components (int): Number of dimensions to project (can be 2 or 3).
        metric_mds (bool): Perform metric (True) or non-metric (False) dimensional scaling. Default True.
        labels (list): Can override labels stored in Adjacency Class.
        labels_color (list): List of colors for labels.
        cmap: Colormap instance (default: ``plt.cm.hot_r``).
        view (tuple): View for 3-Dimensional plot. Default (30, 20).
        figsize (list): Figure size. Default [12, 8].
        ax: Matplotlib axis handle.
        n_jobs (int): Number of parallel jobs.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS

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

    # Run MDS
    mds = MDS(
        n_components=n_components,
        metric=metric_mds,
        n_jobs=n_jobs,
        dissimilarity="precomputed",
        **kwargs,
    )
    proj = mds.fit_transform(adj.squareform())

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
