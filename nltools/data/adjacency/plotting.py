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


def _triangle_pair(value):
    """Split a per-triangle argument into its `(upper, lower)` halves.

    A 2-tuple gives each triangle its own value; anything else, `None`
    included, is one value both triangles share.

    Args:
        value: Scalar for both triangles, or a `(upper, lower)` tuple.

    Returns:
        tuple: The upper and lower value.
    """
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], value[1]
    return value, value


def _stacked_triangle_kwargs(square, mask, cmap, vmin, vmax, kwargs):
    """Resolve one triangle's heatmap keywords, caller's values over the divergent defaults.

    Limits always end up explicit, even when the caller gave none, so the two
    triangles can be compared for a shared colorbar instead of each being
    scaled by seaborn behind our back.

    Args:
        square (np.ndarray): The triangle's source matrix.
        mask (np.ndarray): Boolean mask of the cells this triangle hides.
        cmap: Colormap for this triangle, or None for the default.
        vmin: Lower limit for this triangle, or None to take it from the data.
        vmax: Upper limit for this triangle, or None to take it from the data.
        kwargs (dict): The caller's remaining `seaborn.heatmap` keywords.

    Returns:
        dict: Heatmap keywords for this triangle.
    """
    resolved = dict(kwargs)
    if cmap is not None:
        resolved["cmap"] = cmap
    if vmin is not None:
        resolved["vmin"] = vmin
    if vmax is not None:
        resolved["vmax"] = vmax
    for name, value in _divergent_heatmap_defaults(square).items():
        resolved.setdefault(name, value)
    values = square[~mask]
    if values.size and not np.all(np.isnan(values)):
        resolved.setdefault("vmin", float(np.nanmin(values)))
        resolved.setdefault("vmax", float(np.nanmax(values)))
    return resolved


def _color_scale(heatmap_kwargs):
    """The part of a triangle's keywords a colorbar speaks for."""
    return tuple(
        str(heatmap_kwargs.get(name)) for name in ("cmap", "center", "vmin", "vmax")
    )


def _plot_stacked(
    adj,
    data,
    *,
    labels=None,
    upper_title=None,
    lower_title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    ax=None,
    **kwargs,
):
    """Draw two matrices as the complementary triangles of one square.

    `adj` fills the upper-right triangle and `data` the lower-left, with the
    diagonal hidden in both so a one-cell white gap separates them. Each
    triangle carries its own colormap and limits, resolved from
    `_divergent_heatmap_defaults` unless the caller names them, so the two
    matrices need not share units.

    Args:
        adj (Adjacency): Single matrix drawn in the upper triangle.
        data (Adjacency): Single matrix over the same nodes, drawn in the lower
            triangle.
        labels (list, optional): Node tick labels. Defaults to `adj.labels`, or
            no ticks when it has none; `False` suppresses them.
        upper_title (str, optional): Title drawn above the square.
        lower_title (str, optional): Title drawn below the square.
        cmap (str | matplotlib.colors.Colormap | tuple, optional): One
            colormap for both triangles, or an `(upper, lower)` tuple.
        vmin (float | tuple, optional): One lower limit for both triangles,
            or an `(upper, lower)` tuple.
        vmax (float | tuple, optional): One upper limit for both triangles,
            or an `(upper, lower)` tuple.
        colorbar (bool): Draw colorbars. One bar when the triangles share a
            colormap and limits, two when they do not. Default True.
        ax (matplotlib.axes.Axes, optional): Axis to draw on.
        **kwargs (dict): Forwarded to `seaborn.heatmap` for both triangles;
            `cbar`, `cbar_ax` and `mask` are controlled here.

    Returns:
        matplotlib.axes.Axes: The axis holding both triangles.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from nltools.data import Adjacency

    if not isinstance(data, Adjacency):
        raise ValueError("data must be an Adjacency instance.")
    if not adj.is_single_matrix or not data.is_single_matrix:
        raise ValueError(
            "plot_stacked draws one matrix per triangle; index a stack first."
        )
    if adj.n_nodes != data.n_nodes:
        raise ValueError(
            "Both matrices must describe the same nodes; got "
            f"{adj.n_nodes} and {data.n_nodes}."
        )

    upper_square = adj.squareform()
    lower_square = data.squareform()
    ones = np.ones((adj.n_nodes, adj.n_nodes), dtype=bool)
    upper_mask = np.tril(ones)
    lower_mask = np.triu(ones)

    upper_cmap, lower_cmap = _triangle_pair(cmap)
    upper_vmin, lower_vmin = _triangle_pair(vmin)
    upper_vmax, lower_vmax = _triangle_pair(vmax)
    upper_kwargs = _stacked_triangle_kwargs(
        upper_square, upper_mask, upper_cmap, upper_vmin, upper_vmax, kwargs
    )
    lower_kwargs = _stacked_triangle_kwargs(
        lower_square, lower_mask, lower_cmap, lower_vmin, lower_vmax, kwargs
    )

    if labels is None:
        labels = adj.labels if adj.labels else False
    if labels is not False and len(labels) != adj.n_nodes:
        raise ValueError("labels must have one entry per node.")

    if ax is None:
        _, ax = plt.subplots(1, figsize=(7, 6))
    ax.set_facecolor("white")

    if not colorbar:
        upper_cbar_ax, lower_cbar_ax = None, None
    elif _color_scale(upper_kwargs) == _color_scale(lower_kwargs):
        upper_cbar_ax, lower_cbar_ax = ax.inset_axes([1.03, 0.15, 0.03, 0.7]), None
    else:
        upper_cbar_ax = ax.inset_axes([1.03, 0.55, 0.03, 0.42])
        lower_cbar_ax = ax.inset_axes([1.03, 0.03, 0.03, 0.42])

    for square, mask, triangle_kwargs, cbar_ax in (
        (upper_square, upper_mask, upper_kwargs, upper_cbar_ax),
        (lower_square, lower_mask, lower_kwargs, lower_cbar_ax),
    ):
        triangle_kwargs["mask"] = mask
        triangle_kwargs["ax"] = ax
        triangle_kwargs["cbar"] = cbar_ax is not None
        if cbar_ax is not None:
            triangle_kwargs["cbar_ax"] = cbar_ax
        triangle_kwargs.setdefault("square", True)
        triangle_kwargs.setdefault("linewidths", 0.5)
        triangle_kwargs.setdefault("linecolor", "white")
        triangle_kwargs.setdefault("xticklabels", labels)
        triangle_kwargs.setdefault("yticklabels", labels)
        sns.heatmap(square, **triangle_kwargs)

    if upper_title is not None:
        ax.set_title(upper_title)
    if lower_title is not None:
        ax.set_xlabel(lower_title)
    return ax


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
        # `squeeze=False` keeps a single panel indexable like any other.
        _, a = plt.subplots(nrows=n_subs, figsize=(7, len(adj) * 5), squeeze=False)
        a = a.ravel()
        for i in range(n_subs):
            # Selecting the matrix resolves shared and nested labels alike;
            # `adj.labels[i]` would index a node when the labels are shared.
            matrix = adj[i]
            square = matrix.squareform()
            heatmap_kwargs = _heatmap_kwargs(square, kwargs)
            if matrix.labels:
                sns.heatmap(
                    square,
                    square=True,
                    xticklabels=matrix.labels,
                    yticklabels=matrix.labels,
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
