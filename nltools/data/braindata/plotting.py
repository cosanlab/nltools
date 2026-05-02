"""BrainData plotting functions."""

import os
import warnings

import numpy as np


DEFAULT_SLICE_CUT_COORDS = {
    "x": list(range(-50, 51, 8)),
    "y": list(range(-80, 50, 10)),
    "z": list(range(-40, 71, 9)),
}


def _require_standard_space(bd, op_name: str, *, remedy: str) -> None:
    """Raise if ``bd`` is not in a standard MNI space supported by templates.

    Used to gate plotting paths that draw against MNI-aligned scaffolding
    (glass-brain outlines, fsaverage surfaces, template backgrounds).
    Native-space data would render in misleading positions.
    """
    from nltools.templates import is_standard_space

    ok, reason = is_standard_space(bd.mask.affine)
    if ok:
        return
    raise ValueError(
        f"{op_name} requires data in standard MNI space, but {reason}. {remedy}"
    )


def plot_brain(
    bd,
    method="glass",
    upper=None,
    lower=None,
    threshold=None,
    view="z",
    cut_coords=None,
    cmap=None,
    bg_img=None,
    ax=None,
    figsize=(8, 6),
    title=None,
    colorbar=True,
    save=None,
    stat="mean",
    limit=3,
    **kwargs,
):
    """Plot BrainData instance using nilearn visualization or matplotlib.

    Args:
        bd: BrainData instance.
        method (str): Visualization type ('glass', 'slices', 'timeseries', 'histogram').
        upper (str/float, optional): Upper threshold applied to the data
            (nltools semantics; may be a percentile string like ``"95%"``).
        lower (str/float, optional): Lower threshold applied to the data
            (nltools semantics).
        threshold (float, optional): Absolute-value transparency cutoff
            forwarded to the underlying nilearn plot function. Voxels with
            ``|value| < threshold`` are rendered transparent. Must be >= 0.
            Use ``upper``/``lower`` for one-sided data thresholding.
        view (str): For ``method="slices"``, any non-empty combination of
            ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``).
            Default: ``"z"``.
        cut_coords (list or dict, optional): Cut coordinates for multi-slice
            views. If provided, takes precedence over ``view``-based defaults.
            Either a list of per-axis coordinate sequences whose length
            matches ``view``, or a dict keyed by axis letter (``{"x": [...],
            "z": [...]}``) from which entries for each axis in ``view`` are
            looked up.
        cmap (str, optional): Colormap name.
        bg_img (Nifti1Image or str, optional): Background image for slice views.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
        figsize (tuple, optional): default figure size if no axis (8, 6)
        title (str, optional): Plot title.
        colorbar (bool): Whether to show colorbar. Default: True.
        save (str, optional): Path to save figure(s).
        stat (str): Statistic for timeseries plots. Valid options:
            'mean', 'median', 'std'.
        limit (int): Maximum number of images to render when ``bd`` contains
            multiple maps and ``method`` is ``"glass"`` or ``"slices"``.
            Default: 3. A warning is emitted if the data has more images than
            ``limit``. Ignored for single-image data and for matplotlib-based
            methods (``"timeseries"``, ``"histogram"``), which already
            aggregate across images.
        **kwargs: Additional arguments passed to nilearn plot functions.

    Returns:
        matplotlib.figure.Figure or list[matplotlib.figure.Figure]: For
        single-image data, the figure object (last one created if
        ``method="slices"`` produced multiple per-axis figures). For
        multi-image data with ``method`` in ``{"glass", "slices"}``, a list
        of figures (one per image for glass; one per image-and-view pair for
        slices). All figures auto-display in notebooks.
    """
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_glass_brain, plot_stat_map

    from nltools.templates import get_bg_image

    # Validate inputs
    if bd.is_empty:
        raise ValueError("Cannot plot empty BrainData object")

    if threshold is not None and threshold < 0:
        raise ValueError(
            f"`threshold` is an absolute-value cutoff and must be >= 0 "
            f"(got {threshold}). Use `upper` / `lower` for one-sided data "
            f"thresholding."
        )

    # Validate 'method' parameter
    valid_methods = ["glass", "slices", "timeseries", "histogram"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid 'method' parameter: '{method}'. Must be one of: {valid_methods}. "
        )

    # Handle matplotlib-based plots (timeseries, histogram)
    if method in ["timeseries", "histogram"]:
        return _plot_matplotlib(
            bd, method=method, stat=stat, ax=ax, figsize=figsize, title=title, save=save
        )

    # Parse `view` into an ordered list of axis letters (only matters for
    # method="slices"; cheap to compute so we always do it).
    views = list(view.lower()) if isinstance(view, str) else []
    if not views or not set(views).issubset({"x", "y", "z"}):
        raise ValueError(
            f"Invalid `view`: {view!r}. Must be a non-empty string containing "
            "any combination of 'x', 'y', 'z' (e.g. 'xyz', 'xz', 'y')."
        )

    # Resolve cut_coords against `views`. User-supplied cut_coords take
    # precedence; defaults are drawn from DEFAULT_SLICE_CUT_COORDS per-axis.
    if cut_coords is None:
        cut_coords = [DEFAULT_SLICE_CUT_COORDS[v] for v in views]
    elif isinstance(cut_coords, dict):
        missing = [v for v in views if v not in cut_coords]
        if missing:
            raise ValueError(
                f"`cut_coords` dict is missing entries for axes {missing} "
                f"required by view={view!r}."
            )
        cut_coords = [cut_coords[v] for v in views]
    else:
        cut_coords = [list(c) if isinstance(c, range) else c for c in cut_coords]
        if len(cut_coords) != len(views):
            raise ValueError(
                f"`cut_coords` has {len(cut_coords)} entries but view={view!r} "
                f"requires {len(views)}."
            )

    # Decide which images to plot. For multi-image data we render up to
    # `limit` maps and return a list of figures so the user can see (and
    # programmatically access) each map instead of silently dropping all
    # but the first.
    multi = len(bd.shape) > 1 and bd.shape[0] > 1
    if multi:
        n_total = bd.shape[0]
        n_to_plot = min(n_total, limit)
        if n_total > limit:
            warnings.warn(
                f"BrainData contains {n_total} images; plotting first "
                f"{n_to_plot}. Pass `limit={n_total}` (or higher) to plot "
                "more, or index/aggregate before calling .plot().",
                UserWarning,
                stacklevel=2,
            )
        sub_objs = [bd[i] for i in range(n_to_plot)]
    else:
        sub_objs = [bd]

    # Standard-space gate. Glass brain draws an MNI-shape outline, and the
    # default slice background is looked up from the MNI template registry
    # — both are misleading on native-space data. Slices with a user-
    # supplied bg_img work for any space and are the documented escape
    # hatch (see Miyawaki / native-space tutorials).
    from nltools.templates import is_standard_space

    standard, reason = is_standard_space(bd.mask.affine)
    if not standard:
        if method == "glass":
            if bg_img is not None:
                warnings.warn(
                    f"method='glass' requires standard MNI space ({reason}); "
                    f"falling back to method='slices' with the bg_img you "
                    "provided.",
                    UserWarning,
                    stacklevel=2,
                )
                method = "slices"
            else:
                raise ValueError(
                    f"method='glass' requires data in standard MNI space, "
                    f"but {reason}. Pass method='slices' with "
                    f"bg_img=<your subject anatomical>, or call "
                    f"bd.resample() to bring data into standard space first."
                )
        elif method == "slices" and bg_img is None:
            raise ValueError(
                f"Cannot auto-resolve a background image for non-standard-"
                f"space data ({reason}). Pass "
                f"bg_img=<your subject anatomical>, or call bd.resample() "
                f"to bring data into standard space first."
            )

    # Resolve background image once for slices (template lookup is the same
    # across images sharing a mask).
    if method == "slices" and bg_img is None:
        bg_img = get_bg_image(bd.mask.affine)

    # Collect the matplotlib figure underlying each nilearn display, so the
    # return value has a standard `_repr_*_` path and is recognized by
    # frontend filters. For single-image data the last figure is detached
    # from pyplot below to avoid double-display via `flush_figures`.
    figures = []

    for idx, sub in enumerate(sub_objs):
        # Apply thresholding per-image
        if upper is not None or lower is not None:
            obj = sub.threshold(upper=upper, lower=lower)
        else:
            obj = sub

        cmap_use = cmap if cmap is not None else auto_select_colormap(obj.data)
        save_paths = prepare_save_paths(save, idx if multi else None) if save else None

        try:
            nifti_img = obj.to_nifti()
        except Exception as e:
            raise RuntimeError(f"Failed to convert BrainData to NIfTI: {e}") from e

        # Per-image kwargs. Use the BrainData mask as a transparency image
        # so voxels outside the mask render transparent (nilearn >= 0.12).
        # Users can override by passing their own `transparency=` kwarg.
        plot_kwargs = kwargs.copy()
        plot_kwargs.pop("how", None)
        if multi:
            sub_title = f"{title} (image {idx})" if title else f"image {idx}"
        else:
            sub_title = title
        if sub_title:
            plot_kwargs["title"] = sub_title
        if threshold is not None:
            plot_kwargs["threshold"] = threshold
        plot_kwargs.setdefault("transparency", obj.mask)

        if method == "glass":
            display_glass = plot_glass_brain(
                nifti_img,
                display_mode="lzry",
                colorbar=colorbar,
                cmap=cmap_use,
                plot_abs=False,
                **plot_kwargs,
            )
            fig = display_glass.frame_axes.figure
            if save_paths:
                fig.savefig(save_paths["glass"], bbox_inches="tight")
            figures.append(fig)

        elif method == "slices":
            for v, c in zip(views, cut_coords):
                savefile = save_paths["slices"][v] if save_paths else None
                display_slice = plot_stat_map(
                    nifti_img,
                    cut_coords=c,
                    display_mode=v,
                    cmap=cmap_use,
                    bg_img=bg_img,
                    colorbar=colorbar,
                    **plot_kwargs,
                )
                fig = display_slice.frame_axes.figure
                if savefile:
                    fig.savefig(savefile, bbox_inches="tight")
                figures.append(fig)

    if not figures:
        return None
    if multi:
        # Leave all figures attached to pyplot's tracker so notebook
        # auto-display via `flush_figures` renders each one. Return the
        # list for programmatic access.
        return figures
    # Single image: detach only the figure we return so its `_repr_*_`
    # rendering doesn't duplicate via `flush_figures`. Any earlier per-view
    # figures from method="slices" stay on pyplot's tracker so the cell's
    # post-hook can display them.
    plt.close(figures[-1])
    return figures[-1]


def plot_flatmap_brain(
    bd,
    threshold=None,
    cmap="RdBu_r",
    vmax=None,
    vmin=None,
    template="fsaverage5",
    with_curvature=True,
    curvature_contrast=0.5,
    curvature_brightness=0.5,
    transparency="auto",
    colorbar=True,
    colorbar_orientation="horizontal",
    figsize=(12, 6),
    title=None,
    radius_mm=3.0,
    interpolation="linear",
    axes=None,
    save=None,
):
    """Plot brain data on cortical flatmap.

    Args:
        bd: BrainData instance.
        threshold (float, optional): Values below this absolute threshold
            are masked.
        cmap (str): Matplotlib colormap for data. Default: 'RdBu_r'.
        vmax (float, optional): Maximum value for colormap.
        vmin (float, optional): Minimum value for colormap.
        template (str): fsaverage resolution. Default: 'fsaverage5'.
        with_curvature (bool): Show sulcal/gyral pattern. Default: True.
        curvature_contrast (float): Contrast of curvature. Default: 0.5.
        curvature_brightness (float): Mean brightness of curvature.
            Default: 0.5.
        colorbar (bool): Show colorbar. Default: True.
        colorbar_orientation (str): 'horizontal' or 'vertical'.
            Default: 'horizontal'.
        figsize (tuple): Figure size. Default: (12, 6).
        title (str, optional): Figure title.
        radius_mm (float): sampling radius in mm for vol_to_surf.
            Default: 3.0.
        interpolation (str): Interpolation for vol_to_surf.
            Default: 'linear'.
        axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
        save (str, optional): File path to save figure.

    Returns:
        matplotlib.figure.Figure
    """
    from nltools.plotting import plot_flatmap

    if bd.is_empty:
        raise ValueError("Cannot plot empty BrainData object")

    _require_standard_space(
        bd,
        "plot_flatmap",
        remedy=(
            "Flatmap projection samples vol_to_surf at fsaverage (MNI-"
            "aligned) coordinates and produces garbage on native-space "
            "data. Use bd.plot(method='slices', bg_img=<your subject "
            "anatomical>) instead, or call bd.resample() to bring data "
            "into standard space first."
        ),
    )

    return plot_flatmap(
        brain=bd,
        threshold=threshold,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        template=template,
        with_curvature=with_curvature,
        curvature_contrast=curvature_contrast,
        curvature_brightness=curvature_brightness,
        transparency=transparency,
        colorbar=colorbar,
        colorbar_orientation=colorbar_orientation,
        figsize=figsize,
        title=title,
        radius_mm=radius_mm,
        interpolation=interpolation,
        axes=axes,
        save=save,
    )


def _plot_matplotlib(
    bd, method, stat="mean", figsize=(8, 6), ax=None, title=None, save=None
):
    """Plot using matplotlib (timeseries or histogram).

    Args:
        bd: BrainData instance.
        method (str): 'timeseries' or 'histogram'
        stat (str): Statistic for timeseries ('mean', 'median', 'std')
        figsize (tuple, optional): default figure size if no axis (8, 6)
        ax: Matplotlib axis.
        title (str, optional): Plot title.
        save (str, optional): Path to save figure.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # Create axis if not provided. Track ownership so we only detach figures
    # we created from pyplot's tracker — caller-supplied axes belong to the
    # caller's figure lifecycle.
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        owns_fig = True
    else:
        fig = ax.figure
        owns_fig = False

    if method == "timeseries":
        # For single image, raise informative error
        if len(bd.shape) == 1 or (len(bd.shape) > 1 and bd.shape[0] == 1):
            raise ValueError(
                "timeseries plotting requires multiple images. "
                f"Got {bd.shape[0] if len(bd.shape) > 1 else 1} image(s). "
                "Use histogram for single image visualization."
            )

        # Compute statistic across voxels for each image
        if stat == "mean":
            values = bd.mean(axis=1)
        elif stat == "median":
            values = bd.median(axis=1)
        elif stat == "std":
            values = bd.std(axis=1)
        else:
            raise ValueError(
                f"Invalid stat '{stat}'. Must be 'mean', 'median', or 'std'"
            )

        # Ensure values is 1D array
        if hasattr(values, "data"):
            values = values.data
        values = np.array(values).flatten()

        # Plot
        ax.plot(values, linewidth=2)
        ax.set_xlabel("Image Index", fontsize=12)
        ax.set_ylabel(f"{stat.capitalize()} Across Voxels", fontsize=12)
        if title is None:
            title = f"{stat.capitalize()} Across Voxels"
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

    elif method == "histogram":
        # Flatten data for histogram
        if len(bd.shape) == 1:
            data_flat = bd.data
        else:
            data_flat = bd.data.flatten()

        # Remove NaN/Inf
        data_flat = data_flat[np.isfinite(data_flat)]

        # Plot histogram
        ax.hist(data_flat, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Voxel Value", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        if title is None:
            title = "Voxel Value Distribution"
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

    # Save if requested
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=150)

    if owns_fig:
        plt.close(fig)
    return fig


def auto_select_colormap(data):
    """Auto-select colormap based on data characteristics.

    Args:
        data (np.ndarray): numpy array of brain data

    Returns:
        str: Colormap name
    """
    # Flatten data for analysis
    if data.ndim > 1:
        data_flat = data.flatten()
    else:
        data_flat = data

    # Remove NaN/Inf
    data_flat = data_flat[np.isfinite(data_flat)]

    if len(data_flat) == 0:
        return "RdBu_r"  # Default fallback

    # Check data range for colormap selection
    # If mostly positive (> 90% positive), use hot/reds
    positive_ratio = np.sum(data_flat > 0) / len(data_flat)
    if positive_ratio > 0.9:
        return "hot"
    # If mostly negative (> 90% negative), use cool/blues
    if (1 - positive_ratio) > 0.9:
        return "cool"
    # Otherwise use bipolar
    return "RdBu_r"


def prepare_save_paths(save, idx=None):
    """Prepare save paths for multiple plot outputs.

    Args:
        save: Base save path (str or Path)
        idx (int, optional): Image index appended as ``_img{idx}`` to the
            base filename. Used to disambiguate saves across multiple images.

    Returns:
        dict: Dictionary with 'glass' and 'slices' keys containing save paths
    """
    save = str(save)  # Convert Path objects to strings
    path, filename = os.path.split(save)
    if "." in filename:
        filename, extension = filename.rsplit(".", 1)
    else:
        extension = "png"

    if idx is not None:
        filename = f"{filename}_img{idx}"

    base_path = os.path.join(path, filename) if path else filename

    return {
        "glass": f"{base_path}_glass.{extension}",
        "slices": {
            "x": f"{base_path}_x.{extension}",
            "y": f"{base_path}_y.{extension}",
            "z": f"{base_path}_z.{extension}",
        },
    }
