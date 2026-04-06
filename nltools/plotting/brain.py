"""Brain visualization — surface plots, flatmaps, and interactive viewers."""

__all__ = [
    "plot_interactive_brain",
    "surface_plot",
    "plot_flatmap",
]

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn.plotting import (
    plot_surf_stat_map,
    view_img,
    view_img_on_surf,
)
from nilearn.surface import vol_to_surf

from nltools.utils import attempt_to_import, get_resource_path

# Optional dependencies
ipywidgets = attempt_to_import(
    "ipywidgets",
    name="ipywidgets",
    fromlist=["interact", "fixed", "widgets", "BoundedFloatText", "BoundedIntText"],
)


def plot_interactive_brain(
    brain,
    threshold=1e-6,
    surface=False,
    percentile_threshold=False,
    anatomical=None,
    **kwargs,
):
    """
    This function leverages nilearn's new javascript based brain viewer functions to create interactive plotting functionality.

    Args:
        brain (nltools.BrainData): a BrainData instance of 1d or 2d shape (i.e. 3d or 4d volume)
        threshold (float/str): threshold to initialize the visualization, maybe be a percentile string; default 0
        surface (bool): whether to create a surface-based plot; default False
        percentile_threshold (bool): whether to interpret threshold values as percentiles
        kwargs: optional arguments to nilearn.view_img or nilearn.view_img_on_surf

    Returns:
        interactive brain viewer widget
    """

    if ipywidgets is None:
        raise ImportError(
            "ipywidgets>=5.2.2 is required for interactive plotting. Please install this package manually or install nltools with optional arguments: pip install 'nltools[interactive_plots]'"
        )

    if isinstance(threshold, str):
        if threshold[-1] != "%":
            raise ValueError("Starting threshold provided as string must end in '%'")
        percentile_threshold = True
        warnings.warn(
            "Percentile thresholding ignores brain mask. Results are likely more liberal than you expect (e.g. with non-interactive plotting)!"
        )
        threshold = int(threshold[:-1])

    if len(brain.shape) == 2:
        time_slider = True
        max_idx = brain.shape[0] - 1
    elif len(brain.shape) == 1:
        time_slider = False
    else:
        raise ValueError("BrainData object is not 1d or 2d")

    thresh_box = ipywidgets.widgets.FloatText(value=threshold, description="Threshold")

    if time_slider:
        idx = ipywidgets.widgets.IntSlider(
            min=0,
            max=max_idx,
            step=1,
            value=0,
            orientation="horizontal",
            continuous_update=False,
            description="Volume",
            readout_format="d",
        )
    else:
        idx = ipywidgets.widgets.HTML(
            value="Image is 3D", description="Volume", placeholder=""
        )
    ipywidgets.interact(
        _viewer,
        brain=ipywidgets.fixed(brain),
        thresh=thresh_box,
        idx=idx,
        percentile_threshold=percentile_threshold,
        surface=surface,
        anatomical=ipywidgets.fixed(anatomical),
        **kwargs,
    )


def _viewer(brain, thresh, idx, percentile_threshold, surface, anatomical, **kwargs):
    if thresh == 0:
        thresh = 1e-6
    else:
        if percentile_threshold:
            thresh = str(thresh) + "%"
    if isinstance(idx, int):
        b = brain[idx].to_nifti()
    else:
        b = brain.to_nifti()
    if anatomical:
        bg_img = anatomical
    else:
        bg_img = "MNI152"
    cut_coords = kwargs.get("cut_coords", [0, 0, 0])

    if surface:
        return view_img_on_surf(b, threshold=thresh, **kwargs)
    else:
        return view_img(
            b, bg_img=bg_img, threshold=thresh, cut_coords=cut_coords, **kwargs
        )


def _get_surface_paths():
    """Get paths to included surface files.

    Returns:
        dict: Dictionary with keys for surface meshes and background maps.
            Keys include: 'pial_left', 'pial_right', 'inflated_left',
            'inflated_right', 'midthickness_left', 'midthickness_right',
            'white_left', 'white_right', 'curv_left', 'curv_right',
            'sulc_left', 'sulc_right', 'thickness_left', 'thickness_right'.
    """
    from os.path import join

    resource_path = get_resource_path().rstrip(os.pathsep)
    surfaces_dir = join(resource_path, "surfaces")

    paths = {
        # Surface meshes
        "pial_left": join(surfaces_dir, "sub-colin_hemi-L_pial.surf.gii"),
        "pial_right": join(surfaces_dir, "sub-colin_hemi-R_pial.surf.gii"),
        "inflated_left": join(surfaces_dir, "sub-colin_hemi-L_inflated.surf.gii"),
        "inflated_right": join(surfaces_dir, "sub-colin_hemi-R_inflated.surf.gii"),
        "midthickness_left": join(
            surfaces_dir, "sub-colin_hemi-L_midthickness.surf.gii"
        ),
        "midthickness_right": join(
            surfaces_dir, "sub-colin_hemi-R_midthickness.surf.gii"
        ),
        "white_left": join(surfaces_dir, "sub-colin_hemi-L_white.surf.gii"),
        "white_right": join(surfaces_dir, "sub-colin_hemi-R_white.surf.gii"),
        # Background maps
        "curv_left": join(surfaces_dir, "sub-colin_hemi-L_curv.shape.gii"),
        "curv_right": join(surfaces_dir, "sub-colin_hemi-R_curv.shape.gii"),
        "sulc_left": join(surfaces_dir, "sub-colin_hemi-L_sulc.shape.gii"),
        "sulc_right": join(surfaces_dir, "sub-colin_hemi-R_sulc.shape.gii"),
        "thickness_left": join(surfaces_dir, "sub-colin_hemi-L_thickness.shape.gii"),
        "thickness_right": join(surfaces_dir, "sub-colin_hemi-R_thickness.shape.gii"),
    }

    return paths


def _resolve_brain_input(brain):
    """Convert various input types to nibabel Nifti1Image.

    Args:
        brain: BrainData, nibabel Nifti1Image, or file path to NIfTI image.

    Returns:
        nibabel.Nifti1Image: Nifti image object.

    Raises:
        ValueError: If input cannot be converted to Nifti1Image (e.g., empty
            BrainData or file not found).
        TypeError: If input type is not supported.
    """
    import nibabel as nib
    from nltools.data import BrainData

    if isinstance(brain, BrainData):
        if len(brain) == 0:
            raise ValueError("Cannot plot empty BrainData object")
        # If multiple images, use first one
        if len(brain.shape) == 2 and brain.shape[0] > 1:
            brain = brain[0]
        return brain.to_nifti()
    elif isinstance(brain, nib.Nifti1Image):
        return brain
    elif isinstance(brain, (str, Path)):
        if not os.path.exists(brain):
            raise ValueError(f"File not found: {brain}")
        return nib.load(brain)
    else:
        raise TypeError(
            f"Input must be BrainData, nibabel Nifti1Image, or file path, got {type(brain)}"
        )


def _get_background_map(bg_map, hemi):
    """Resolve background map path.

    Args:
        bg_map (str or None): Background map type ('curvature', 'sulc', None,
            or file path to custom background map).
        hemi (str): Hemisphere ('left' or 'right').

    Returns:
        str or None: Path to background map file, or None if bg_map is None.

    Raises:
        ValueError: If bg_map is not recognized or file not found.
    """
    if bg_map is None:
        return None

    if isinstance(bg_map, str) and os.path.exists(bg_map):
        # User provided a file path
        return bg_map

    paths = _get_surface_paths()

    # Map common names to file keys
    bg_map_map = {
        "curvature": "curv",
        "curv": "curv",
        "sulc": "sulc",
        "sulcal": "sulc",
        "thickness": "thickness",
    }

    bg_key = bg_map_map.get(bg_map.lower() if isinstance(bg_map, str) else None)
    if bg_key is None:
        raise ValueError(
            f"Unknown background map: {bg_map}. "
            f"Supported options: {list(bg_map_map.keys())}, None, or file path"
        )

    key = f"{bg_key}_{hemi}"
    if key not in paths:
        raise ValueError(f"Background map not found: {key}")

    return paths[key]


def surface_plot(
    brain,
    surface="inflated",
    bg_map="curvature",
    hemi="both",
    view="montage",
    threshold=None,
    cmap="RdBu_r",
    vmax=None,
    vmin=None,
    darkness=None,
    bg_on_data=False,
    colorbar=False,
    figsize=(10, 10),
    n_samples=1,
    radius=0.0,
    interpolation="linear",
    engine="matplotlib",
    axes=None,
    save=None,
    **kwargs,
):
    """Plot neuroimaging data on cortical surface.

    Intelligently projects volumetric NIfTI data onto cortical surfaces
    and displays in customizable montage layouts. Automatically handles
    hemispheric parsing and uses included MNI152 template surfaces.

    Args:
        brain: BrainData, nibabel Nifti1Image, or file path to NIfTI image.
            If BrainData has multiple images, plots the first one.
        surface (str, optional): Surface mesh type. Options: 'pial',
            'inflated', 'midthickness', 'white'. Defaults to 'inflated'.
        bg_map (str or None, optional): Background map. Options: 'curvature',
            'sulc', None, or file path to custom background map.
            Defaults to 'curvature'.
        hemi (str, optional): Hemisphere to plot. Options: 'left', 'right',
            'both'. Defaults to 'both'.
        view (str or list, optional): View type. Options: 'lateral', 'medial',
            'montage', or list of views. Defaults to 'montage' (2x2 grid).
        threshold (float or str, optional): Threshold value. Can be a float
            or percentile string like '95%'. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to 'RdBu_r'.
        vmax (float, optional): Maximum value for colormap scaling.
        vmin (float, optional): Minimum value for colormap scaling.
        darkness (float or None, optional): Background darkness (0-1). Defaults to None.
        bg_on_data (bool, optional): Overlay background on data. Defaults to False.
        colorbar (bool, optional): Show colorbar. Defaults to False.
        figsize (tuple, optional): Figure size tuple (width, height).
            Defaults to (10, 10).
        n_samples (int, optional): Number of samples for vol_to_surf projection.
            Defaults to 1.
        radius (float, optional): Sampling radius for vol_to_surf projection.
            Defaults to 0.0.
        interpolation (str, optional): Interpolation method for projection.
            Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'.
        engine (str, optional): Rendering engine. Options: 'matplotlib',
            'plotly'. Defaults to 'matplotlib'.
        axes (matplotlib.axes.Axes or list, optional): Custom matplotlib axes
            for montage layout. If None, creates new figure. Defaults to None.
        save (str or None, optional): File path to save plot. If None, plot
            is displayed but not saved. Defaults to None.
        **kwargs: Additional arguments passed to plot_surf_stat_map.

    Returns:
        matplotlib.figure.Figure or plotly.graph_objects.Figure: Figure object
        containing the surface plot(s).

    Raises:
        ValueError: If input is empty BrainData, invalid surface/view/hemi,
            or surface files not found.
        TypeError: If input type is not supported.

    Examples:
        Plot BrainData with default 2x2 montage:

        >>> from nltools.plotting import surface_plot
        >>> from nltools.data import BrainData
        >>> brain = BrainData('data.nii.gz')
        >>> fig = surface_plot(brain)

        Single hemisphere, lateral view:

        >>> fig = surface_plot(brain, hemi='left', view='lateral')

        Custom colormap and threshold:

        >>> fig = surface_plot(brain, cmap='hot', threshold=0.5)

        Percentile threshold with custom background:

        >>> fig = surface_plot(brain, threshold='95%', bg_map='sulc')
    """
    # Resolve input to nibabel image
    nifti_img = _resolve_brain_input(brain)

    # Get surface paths
    paths = _get_surface_paths()

    # Validate surface type
    valid_surfaces = ["pial", "inflated", "midthickness", "white"]
    if surface not in valid_surfaces:
        raise ValueError(
            f"Invalid surface type: {surface}. Must be one of {valid_surfaces}"
        )

    # Determine which hemispheres to plot
    if hemi == "both":
        hemispheres = ["left", "right"]
    elif hemi in ["left", "right"]:
        hemispheres = [hemi]
    else:
        raise ValueError(f"Invalid hemi: {hemi}. Must be 'left', 'right', or 'both'")

    # Determine views
    if view == "montage":
        views = ["lateral", "medial"]
    elif isinstance(view, str):
        views = [view]
    elif isinstance(view, list):
        views = view
    else:
        raise ValueError(
            f"Invalid view: {view}. Must be 'lateral', 'medial', 'montage', or list"
        )

    # Validate views
    valid_views = ["lateral", "medial"]
    for v in views:
        if v not in valid_views:
            raise ValueError(f"Invalid view: {v}. Must be one of {valid_views}")

    # Project volume to surface textures for each hemisphere
    textures = {}
    for h in hemispheres:
        surface_key = f"{surface}_{h}"
        if surface_key not in paths:
            raise ValueError(f"Surface file not found: {surface_key}")

        textures[h] = vol_to_surf(
            nifti_img,
            paths[surface_key],
            interpolation=interpolation,
            n_samples=n_samples,
            radius=radius,
        )

    # Prepare background maps
    bg_maps = {}
    for h in hemispheres:
        bg_maps[h] = _get_background_map(bg_map, h)

    # Handle threshold (percentile string)
    if isinstance(threshold, str) and threshold.endswith("%"):
        # Convert percentile to actual threshold value
        percentile = float(threshold[:-1])
        all_values = np.concatenate([textures[h] for h in hemispheres])
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            threshold = np.percentile(np.abs(all_values), percentile)

    # Create figure and axes if not provided
    if axes is None:
        if hemi == "both" and view == "montage":
            # Default 2x2 montage: LH lateral, RH lateral, LH medial, RH medial
            fig, axes = plt.subplots(
                2, 2, figsize=figsize, subplot_kw={"projection": "3d"}
            )
            axes = axes.flatten()
        elif len(hemispheres) == 1 and len(views) == 1:
            # Single plot
            fig, axes = plt.subplots(
                1, 1, figsize=figsize, subplot_kw={"projection": "3d"}
            )
            axes = [axes]
        else:
            # Custom layout
            n_plots = len(hemispheres) * len(views)
            fig, axes = plt.subplots(
                1, n_plots, figsize=figsize, subplot_kw={"projection": "3d"}
            )
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
    else:
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        elif not isinstance(axes, list):
            axes = [axes]
        fig = axes[0].figure

    # Plot each combination
    plot_idx = 0
    for view_name in views:
        for h in hemispheres:
            if plot_idx >= len(axes):
                break

            surface_key = f"{surface}_{h}"
            mesh = paths[surface_key]
            texture = textures[h]
            bg = bg_maps[h]

            # Plot on specified axis
            plot_surf_stat_map(
                mesh,
                texture,
                hemi=h,
                view=view_name,
                bg_map=bg,
                bg_on_data=bg_on_data,
                darkness=darkness,
                colorbar=colorbar,
                cmap=cmap,
                threshold=threshold,
                vmax=vmax,
                vmin=vmin,
                axes=axes[plot_idx],
                engine=engine,
                **kwargs,
            )
            plot_idx += 1

    # Adjust layout
    if hemi == "both" and view == "montage":
        plt.subplots_adjust(wspace=-0.05, hspace=-0.1)

    # Save if requested
    if save is not None:
        fig.savefig(save, bbox_inches="tight", transparent=True, dpi=300)

    return fig


def plot_flatmap(
    brain,
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
    radius=3.0,
    interpolation="linear",
    axes=None,
    save=None,
):
    """Plot brain data on cortical flatmap.

    Projects MNI152 volumetric data onto an fsaverage surface and renders
    as a 2D flattened cortical map. Uses nilearn's vol_to_surf for projection
    and matplotlib's tripcolor for rendering.

    This function provides publication-quality flatmap visualizations without
    requiring external dependencies like pycortex.

    Args:
        brain: BrainData, nibabel Nifti1Image, or file path to NIfTI image.
            Data must be in MNI152 space.
        threshold (float or str, optional): Values below this absolute
            threshold are masked. Can be a float or percentile string
            like '95%'. Defaults to None (no threshold).
        cmap (str, optional): Matplotlib colormap for data. Defaults to
            'RdBu_r' (diverging red-blue).
        vmax (float, optional): Maximum value for colormap. If None,
            uses symmetric max of absolute values.
        vmin (float, optional): Minimum value for colormap. If None
            and vmax is set, uses -vmax for diverging maps.
        template (str, optional): fsaverage resolution. Options:
            'fsaverage3' (642 vertices), 'fsaverage4' (2562),
            'fsaverage5' (10242, default), 'fsaverage6' (40962),
            'fsaverage' (163842, full resolution).
        with_curvature (bool, optional): Show sulcal/gyral pattern as
            grayscale background. Defaults to True.
        curvature_contrast (float, optional): Contrast of curvature
            (0=flat gray, 1=full contrast). Defaults to 0.5.
        curvature_brightness (float, optional): Mean brightness of
            curvature (0=dark, 1=bright). Defaults to 0.5.
        colorbar (bool, optional): Show colorbar. Defaults to True.
        colorbar_orientation (str, optional): 'horizontal' or 'vertical'.
            Defaults to 'horizontal'.
        figsize (tuple, optional): Figure size (width, height).
            Defaults to (12, 6).
        title (str, optional): Figure title. Defaults to None.
        radius (float, optional): Sampling radius in mm for vol_to_surf
            projection. Larger values provide smoother projections.
            Defaults to 3.0.
        interpolation (str, optional): Interpolation for vol_to_surf.
            Options: 'linear', 'nearest_most_frequent'. Defaults to 'linear'.
        axes (matplotlib.axes.Axes, optional): Existing axes to plot on.
            If None, creates new figure. Defaults to None.
        save (str, optional): File path to save figure. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure containing the flatmap.

    Examples:
        Basic flatmap with default settings:

        >>> from nltools.plotting import plot_flatmap
        >>> from nltools.data import BrainData
        >>> brain = BrainData('stats.nii.gz')
        >>> fig = plot_flatmap(brain)

        Thresholded with custom colormap:

        >>> fig = plot_flatmap(brain, threshold=2.5, cmap='hot')

        Percentile threshold, no curvature:

        >>> fig = plot_flatmap(brain, threshold='95%', with_curvature=False)

        High resolution for publication:

        >>> fig = plot_flatmap(brain, template='fsaverage6', figsize=(16, 8))
        >>> fig.savefig('flatmap.pdf', dpi=300)

    Notes:
        - Data is projected from MNI152 space to fsaverage surface space.
          Small alignment differences are expected at boundaries.
        - Higher resolution templates (fsaverage6, fsaverage) produce
          sharper images but take longer to render.
        - The flat surfaces are cached by nilearn after first download
          (~50MB for fsaverage5).
    """
    from nilearn import datasets, surface
    import nibabel as nib
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Resolve input to nibabel image
    nifti_img = _resolve_brain_input(brain)

    # Fetch fsaverage surfaces (cached after first download)
    fs = datasets.fetch_surf_fsaverage(template)

    # Project volume to surface for both hemispheres
    texture_left = surface.vol_to_surf(
        nifti_img,
        fs["pial_left"],
        radius=radius,
        interpolation=interpolation,
    )
    texture_right = surface.vol_to_surf(
        nifti_img,
        fs["pial_right"],
        radius=radius,
        interpolation=interpolation,
    )

    # Load flat surface meshes
    flat_left = nib.load(fs["flat_left"])
    flat_right = nib.load(fs["flat_right"])

    coords_left = flat_left.darrays[0].data[:, :2]  # Only X, Y for flatmap
    coords_right = flat_right.darrays[0].data[:, :2]
    faces_left = flat_left.darrays[1].data
    faces_right = flat_right.darrays[1].data

    # Offset right hemisphere to the right of left hemisphere
    gap = 20  # Gap between hemispheres in surface units
    coords_right = coords_right.copy()
    coords_right[:, 0] += coords_left[:, 0].max() - coords_right[:, 0].min() + gap

    # Load curvature for background
    if with_curvature:
        curv_left = nib.load(fs["curv_left"]).darrays[0].data
        curv_right = nib.load(fs["curv_right"]).darrays[0].data

    # Handle threshold (percentile string)
    if isinstance(threshold, str) and threshold.endswith("%"):
        percentile = float(threshold[:-1])
        all_values = np.concatenate([texture_left, texture_right])
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            threshold = np.percentile(np.abs(all_values), percentile)

    # Determine colormap range
    if vmax is None:
        all_values = np.concatenate([texture_left, texture_right])
        vmax = np.nanmax(np.abs(all_values))
    if vmin is None:
        vmin = -vmax  # Symmetric for diverging colormaps

    # Apply threshold masking
    if threshold is not None:
        texture_left_masked = np.where(
            np.abs(texture_left) >= threshold, texture_left, np.nan
        )
        texture_right_masked = np.where(
            np.abs(texture_right) >= threshold, texture_right, np.nan
        )
    else:
        texture_left_masked = texture_left
        texture_right_masked = texture_right

    # Create figure and axes
    if axes is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        ax = axes
        fig = ax.figure

    # Plot curvature background
    if with_curvature:
        # Normalize and adjust curvature for display
        curv_norm = Normalize(vmin=-0.5, vmax=0.5)

        # Scale by contrast and shift by brightness
        curv_left_display = (curv_norm(curv_left) - 0.5) * curvature_contrast
        curv_left_display = curv_left_display + curvature_brightness
        curv_right_display = (curv_norm(curv_right) - 0.5) * curvature_contrast
        curv_right_display = curv_right_display + curvature_brightness

        # Plot curvature as background (zorder=0)
        ax.tripcolor(
            coords_left[:, 0],
            coords_left[:, 1],
            faces_left,
            curv_left_display,
            cmap="gray",
            shading="gouraud",
            vmin=0,
            vmax=1,
            zorder=0,
        )
        ax.tripcolor(
            coords_right[:, 0],
            coords_right[:, 1],
            faces_right,
            curv_right_display,
            cmap="gray",
            shading="gouraud",
            vmin=0,
            vmax=1,
            zorder=0,
        )

    # Plot data overlay
    ax.tripcolor(
        coords_left[:, 0],
        coords_left[:, 1],
        faces_left,
        texture_left_masked,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    ax.tripcolor(
        coords_right[:, 0],
        coords_right[:, 1],
        faces_right,
        texture_right_masked,
        cmap=cmap,
        shading="gouraud",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    # Clean up axes
    ax.set_aspect("equal")
    ax.axis("off")

    # Add title
    if title is not None:
        ax.set_title(title, fontsize=14)

    # Add colorbar
    if colorbar:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin, vmax))
        sm.set_array([])

        if colorbar_orientation == "horizontal":
            fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
        else:
            fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save if requested
    if save is not None:
        fig.savefig(save, bbox_inches="tight", facecolor="white", dpi=300)

    return fig
