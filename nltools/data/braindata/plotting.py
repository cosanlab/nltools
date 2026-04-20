"""BrainData plotting functions."""

import os
import warnings

import numpy as np


def plot_brain(
    bd,
    method="glass",
    upper=None,
    lower=None,
    threshold=None,
    cut_coords=None,
    cmap=None,
    bg_img=None,
    ax=None,
    title=None,
    colorbar=True,
    save=None,
    stat="mean",
    **kwargs,
):
    """Plot BrainData instance using nilearn visualization or matplotlib.

    Args:
        bd: BrainData instance.
        method (str): Visualization type ('glass', 'slices', 'timeseries', 'histogram').
        upper (str/float, optional): Upper threshold.
        lower (str/float, optional): Lower threshold.
        threshold (float, optional): Convenience parameter. If positive,
            sets upper (shows values above threshold). If negative,
            sets lower (shows values below threshold).
        cut_coords (list, optional): Cut coordinates for multi-slice views.
        cmap (str, optional): Colormap name.
        bg_img (Nifti1Image or str, optional): Background image for slice views.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
        title (str, optional): Plot title.
        colorbar (bool): Whether to show colorbar. Default: True.
        save (str, optional): Path to save figure(s).
        stat (str): Statistic for timeseries plots. Valid options:
            'mean', 'median', 'std'.
        **kwargs: Additional arguments passed to nilearn plot functions.

    Returns:
        Display or matplotlib Figure.
    """
    from nilearn.plotting import plot_glass_brain, plot_stat_map
    from nltools.templates import get_bg_image, get_brainspace
    import matplotlib.pyplot as plt

    # Validate inputs
    if bd.is_empty:
        raise ValueError("Cannot plot empty BrainData object")

    # Handle convenience threshold parameter
    if threshold is not None:
        if threshold >= 0:
            upper = threshold
        else:
            lower = threshold

    # Validate 'method' parameter
    valid_methods = ["glass", "slices", "timeseries", "histogram"]
    if method not in valid_methods:
        raise ValueError(
            f"Invalid 'method' parameter: '{method}'. Must be one of: {valid_methods}. "
        )

    # Handle matplotlib-based plots (timeseries, histogram)
    if method in ["timeseries", "histogram"]:
        return _plot_matplotlib(
            bd, method=method, stat=stat, ax=ax, title=title, save=save
        )

    # Handle thresholding
    if upper or lower:
        obj = bd.threshold(upper=upper, lower=lower)
    else:
        obj = bd

    # Ensure single image for plotting
    if len(obj.shape) > 1 and obj.shape[0] > 1:
        obj = obj[0]

    # Default cut coordinates
    if cut_coords is None:
        cut_coords = [
            range(-50, 51, 8),  # x coordinates
            range(-80, 50, 10),  # y coordinates
            range(-40, 71, 9),  # z coordinates
        ]

    # Default colormap with auto-selection
    if cmap is None:
        cmap = auto_select_colormap(obj.data)

    # Views for multi-slice plotting
    views = ["x", "y", "z"]

    # Handle save paths
    save_paths = prepare_save_paths(save) if save else None

    # Convert to nifti
    try:
        nifti_img = obj.to_nifti()
    except Exception as e:
        raise RuntimeError(f"Failed to convert BrainData to NIfTI: {e}") from e

    # Plot based on 'method' parameter
    display_objects = []

    # Prepare kwargs with title if provided
    # Remove 'how' from kwargs if present (backward compatibility)
    plot_kwargs = kwargs.copy()
    plot_kwargs.pop("how", None)  # Remove 'how' if accidentally passed
    if title:
        plot_kwargs["title"] = title

    if method == "glass":
        display_glass = plot_glass_brain(
            nifti_img,
            display_mode="lzry",
            colorbar=colorbar,
            cmap=cmap,
            plot_abs=False,
            **plot_kwargs,
        )
        display_objects.append(display_glass)
        if save_paths:
            plt.savefig(save_paths["glass"], bbox_inches="tight")

    elif method == "slices":
        # Background image selection (respects current brain space)
        if bg_img is None:
            try:
                bg_img = get_bg_image(obj.mask.affine)
            except ValueError as e:
                # Handle non-isometric voxels gracefully
                if "isotropic" in str(e).lower() or "isometric" in str(e).lower():
                    cfg = get_brainspace()
                    warnings.warn(
                        f"Non-isometric voxels detected: {str(e)}. "
                        f"Using default MNI152 template ({cfg.template}, "
                        f"{cfg.resolution}mm) as background image. "
                        f"To use a custom background, provide bg_img parameter.",
                        UserWarning,
                        stacklevel=2,
                    )
                    bg_img = cfg.brain
                else:
                    # Re-raise if it's a different ValueError
                    raise
        for v, c, savefile in zip(
            views, cut_coords, save_paths["slices"] if save_paths else [None] * 3
        ):
            display_slice = plot_stat_map(
                nifti_img,
                cut_coords=c,
                display_mode=v,
                cmap=cmap,
                bg_img=bg_img,
                colorbar=colorbar,
                **plot_kwargs,
            )
            display_objects.append(display_slice)
            if savefile:
                plt.savefig(savefile, bbox_inches="tight")

    # Return last display object or None
    return display_objects[-1] if display_objects else None


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
        colorbar=colorbar,
        colorbar_orientation=colorbar_orientation,
        figsize=figsize,
        title=title,
        radius_mm=radius_mm,
        interpolation=interpolation,
        axes=axes,
        save=save,
    )


def _plot_matplotlib(bd, method, stat="mean", ax=None, title=None, save=None):
    """Plot using matplotlib (timeseries or histogram).

    Args:
        bd: BrainData instance.
        method (str): 'timeseries' or 'histogram'
        stat (str): Statistic for timeseries ('mean', 'median', 'std')
        ax: Matplotlib axis.
        title (str, optional): Plot title.
        save (str, optional): Path to save figure.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

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
    elif (1 - positive_ratio) > 0.9:
        return "cool"
    # Otherwise use bipolar
    else:
        return "RdBu_r"


def prepare_save_paths(save):
    """Prepare save paths for multiple plot outputs.

    Args:
        save: Base save path (str or Path)

    Returns:
        dict: Dictionary with 'glass' and 'slices' keys containing save paths
    """
    save = str(save)  # Convert Path objects to strings
    path, filename = os.path.split(save)
    if "." in filename:
        filename, extension = filename.rsplit(".", 1)
    else:
        extension = "png"

    base_path = os.path.join(path, filename) if path else filename

    return {
        "glass": f"{base_path}_glass.{extension}",
        "slices": [
            f"{base_path}_x.{extension}",
            f"{base_path}_y.{extension}",
            f"{base_path}_z.{extension}",
        ],
    }
