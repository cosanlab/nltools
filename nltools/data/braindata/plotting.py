"""Glass-brain, slice, flatmap, timeseries, and histogram plots for `BrainData`."""

import os
import warnings

import numpy as np

from nltools.utils import find_stack_level
from .utils import _result_from_array


DEFAULT_SLICE_CUT_COORDS = {
    "x": list(range(-50, 51, 8)),
    "y": list(range(-80, 50, 10)),
    "z": list(range(-40, 71, 9)),
}


def _image_world_bounds(nifti_img, axis_letter: str) -> tuple[float, float]:
    """World-coord (lo, hi) bounds of ``nifti_img`` along ``axis_letter``.

    Used to bounds-trim ``DEFAULT_SLICE_CUT_COORDS`` so MNI-shaped defaults
    don't trip nilearn's strict cut_coords validation when the data has
    a non-MNI affine or covers a small native-space FOV.
    """
    from nibabel.affines import apply_affine

    shape = nifti_img.shape[:3]
    affine = nifti_img.affine
    axis_idx = "xyz".index(axis_letter)
    corners = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ]
    )
    world = apply_affine(affine, corners)
    return float(world[:, axis_idx].min()), float(world[:, axis_idx].max())


def plot_brain(
    bd,
    *,
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
        bd (BrainData): Data to plot.
        method (str): Visualization type ('glass', 'slices', 'timeseries', 'histogram').
        upper (str | float | None): Upper threshold applied to the data
            (nltools semantics; may be a percentile string like ``"95%"``).
        lower (str | float | None): Lower threshold applied to the data
            (nltools semantics).
        threshold (float | str, optional): Absolute-value transparency cutoff
            forwarded to nilearn. Percentile strings such as ``"95%"`` are
            resolved over finite, nonzero magnitudes. Must be >= 0.
        view (str): For ``method="slices"``, any non-empty combination of
            ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``).
            Default: ``"z"``.
        cut_coords (list or dict, optional): Cut coordinates for multi-slice
            views. If provided, takes precedence over ``view``-based defaults.
            Either a list of per-axis coordinate sequences whose length
            matches ``view``, or a dict keyed by axis letter (``{"x": [...],
            "z": [...]}``) from which entries for each axis in ``view`` are
            looked up.
        cmap (str, optional): Colormap name. By default, positive-only maps use
            ``"Reds"``, negative-only maps use ``"Blues_r"``, and mixed maps
            use ``"RdBu_r"``.
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
        **kwargs (dict): Additional arguments forwarded to
            `nilearn.plotting.plot_glass_brain` / `plot_stat_map`.

    Returns:
        matplotlib.figure.Figure | list[matplotlib.figure.Figure]: For
            single-image data, the figure object (last one created if
            `method="slices"` produced multiple per-axis figures). For
            multi-image data with `method` in `{"glass", "slices"}`, a list of
            figures (one per image for glass; one per image-and-view pair for
            slices). All figures auto-display in notebooks.
    """
    import matplotlib.pyplot as plt
    from nilearn.plotting import plot_glass_brain, plot_stat_map

    from nltools.templates import get_bg_image

    # Validate inputs
    if bd.is_empty:
        raise ValueError("Cannot plot empty BrainData object")

    if threshold is not None and not isinstance(threshold, str) and threshold < 0:
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
    # Track whether we picked the defaults so we can bounds-trim them later
    # against the actual image — MNI-shaped defaults land outside the data
    # for native-space or small synthetic maps, and nilearn 0.12 rejects
    # all-out-of-bounds cut_coords with an opaque ValueError.
    cut_coords_were_defaulted = cut_coords is None
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
                stacklevel=find_stack_level(),
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
                    stacklevel=find_stack_level(),
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

        from .utils import resolve_threshold

        threshold_use = resolve_threshold(threshold, np.abs(obj.data))
        if threshold_use is not None and threshold_use < 0:
            raise ValueError(
                f"`threshold` is an absolute-value cutoff and must be >= 0 "
                f"(got {threshold_use}). Use `upper` / `lower` for one-sided "
                f"data thresholding."
            )
        displayed_data = obj.data
        if threshold_use is not None:
            displayed_data = displayed_data[np.abs(displayed_data) >= threshold_use]
        cmap_use = cmap if cmap is not None else auto_select_colormap(displayed_data)
        save_paths = prepare_save_paths(save, idx if multi else None) if save else None

        # A plot cannot show NaN/inf; nilearn zero-fills them itself but warns
        # every time, which is noise for ROI maps and tSNR (NaN outside parcels
        # or where std == 0). Zero-fill up front so the result is identical
        # and silent.
        if not np.all(np.isfinite(obj.data)):
            obj = _result_from_array(
                obj,
                np.nan_to_num(obj.data, nan=0.0, posinf=0.0, neginf=0.0),
                rows="preserve",
            )

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
        if threshold_use is not None:
            plot_kwargs["threshold"] = threshold_use
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
                # Trim defaulted MNI cut_coords down to coords that actually
                # land inside the image. If nothing remains (small native-
                # space FOV, single-slice synthetic data), pass None so
                # nilearn picks its own coords within the bounds.
                c_eff = c
                if cut_coords_were_defaulted and isinstance(c_eff, list):
                    lo, hi = _image_world_bounds(nifti_img, v)
                    in_bounds = [float(x) for x in c_eff if lo <= float(x) <= hi]
                    c_eff = in_bounds if in_bounds else None
                display_slice = plot_stat_map(
                    nifti_img,
                    cut_coords=c_eff,
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


def _plot_matplotlib(
    bd, method, stat="mean", figsize=(8, 6), ax=None, title=None, save=None
):
    """Plot using matplotlib (timeseries or histogram).

    Args:
        bd (BrainData): Data to plot.
        method (str): 'timeseries' or 'histogram'.
        stat (str): Statistic for timeseries ('mean', 'median', 'std').
        figsize (tuple): Figure size when no axis is given. Default: (8, 6).
        ax (matplotlib.axes.Axes | None): Existing axis to plot on.
        title (str | None): Plot title.
        save (str | None): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The rendered figure.
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
        data (np.ndarray): Brain data values.

    Returns:
        str: ``'Reds'`` for positive-only data, ``'Blues_r'`` for
            negative-only data, otherwise ``'RdBu_r'``.
    """
    # Flatten data for analysis
    if data.ndim > 1:
        data_flat = data.flatten()
    else:
        data_flat = data

    # Stored zeros are background, not evidence that a map is mixed-signed.
    data_flat = data_flat[np.isfinite(data_flat) & (data_flat != 0)]

    if len(data_flat) == 0:
        return "RdBu_r"  # Default fallback

    if np.all(data_flat > 0):
        return "Reds"
    if np.all(data_flat < 0):
        return "Blues_r"
    return "RdBu_r"


def prepare_save_paths(save, idx=None):
    """Prepare save paths for multiple plot outputs.

    Args:
        save (str | Path): Base save path; its extension is reused (default
            `png`).
        idx (int | None): Image index appended as ``_img{idx}`` to the base
            filename, to disambiguate saves across multiple images.

    Returns:
        dict: `'glass'` maps to one path; `'slices'` maps to a dict of per-axis
            (`'x'`, `'y'`, `'z'`) paths.
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
