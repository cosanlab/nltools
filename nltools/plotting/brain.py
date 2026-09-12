"""Brain visualization — surface plots, flatmaps, and interactive viewers."""

__all__ = [
    "plot_flatmap",
    "plot_interactive_brain",
    "plot_surf",
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

from nltools.utils import attempt_to_import, find_stack_level

# Optional dependencies
ipywidgets = attempt_to_import(
    "ipywidgets",
    fromlist=["interact", "fixed", "widgets", "BoundedFloatText", "BoundedIntText"],
)


def _resolve_stat_map_defaults(data, *, cmap=None, vmin=None, vmax=None):
    """Resolve sign-aware stat-map defaults from finite, nonzero values.

    Explicit values are preserved. Missing range endpoints follow nilearn's
    convention: one-sided maps include zero, while mixed maps are symmetric.
    """
    values = np.asarray(data, dtype=float).ravel()
    values = values[np.isfinite(values) & (values != 0)]

    has_positive = bool(np.any(values > 0))
    has_negative = bool(np.any(values < 0))
    if has_positive and not has_negative:
        default_cmap = "Reds"
        default_vmin = 0.0
        default_vmax = float(values.max())
    elif has_negative and not has_positive:
        default_cmap = "Blues_r"
        default_vmin = float(values.min())
        default_vmax = 0.0
    else:
        default_cmap = "RdBu_r"
        max_abs = float(np.max(np.abs(values))) if values.size else 1.0
        default_vmin = -max_abs
        default_vmax = max_abs

    return (
        default_cmap if cmap is None else cmap,
        default_vmin if vmin is None else vmin,
        default_vmax if vmax is None else vmax,
    )


def plot_interactive_brain(
    brain,
    *,
    threshold=1e-6,
    surface=False,
    percentile_threshold=False,
    anatomical=None,
    **kwargs,
):
    """Create an interactive brain visualization with nilearn.

    Args:
        brain (BrainData): A 1-D (single volume) or 2-D (stack of volumes) instance.
        threshold (float | str): Initial threshold; a percentile string such as
            `'95%'` switches on `percentile_threshold`. Default 1e-6.
        surface (bool): Whether to create a surface-based plot. Default False.
        percentile_threshold (bool): Whether to interpret threshold values as
            percentiles. Default False.
        anatomical (nibabel.Nifti1Image | str, optional): Background image; defaults
            to nilearn's MNI152 template.
        **kwargs (dict): Forwarded to `nilearn.plotting.view_img` or
            `nilearn.plotting.view_img_on_surf`.

    Note:
        Returns nothing; the widgets render inline.
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
            "Percentile thresholding ignores brain mask. Results are likely more liberal than you expect (e.g. with non-interactive plotting)!",
            UserWarning,
            stacklevel=find_stack_level(),
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
    return view_img(b, bg_img=bg_img, threshold=thresh, cut_coords=cut_coords, **kwargs)


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
    if isinstance(brain, nib.Nifti1Image):
        return brain
    if isinstance(brain, (str, Path)):
        if not os.path.exists(brain):
            raise ValueError(f"File not found: {brain}")
        return nib.load(brain)
    raise TypeError(
        f"Input must be BrainData, nibabel Nifti1Image, or file path, got {type(brain)}"
    )


def _resolve_transparency(transparency, brain):
    """Resolve a transparency mask spec to a nibabel Nifti1Image (or None).

    Args:
        transparency: ``"auto"`` (use ``brain.mask`` if ``brain`` is a
            BrainData, else ``None``), ``None`` (no masking), a
            ``BrainData``, a ``nibabel.Nifti1Image``, or a file path.
        brain: The primary ``brain`` argument passed to the plotting
            function; used only to retrieve ``.mask`` when
            ``transparency == "auto"``.

    Returns:
        nibabel.Nifti1Image or None.
    """
    import nibabel as nib
    from nltools.data import BrainData

    if transparency == "auto":
        return brain.mask if isinstance(brain, BrainData) else None
    if transparency is None:
        return None
    if isinstance(transparency, BrainData):
        return transparency.to_nifti()
    if isinstance(transparency, nib.Nifti1Image):
        return transparency
    if isinstance(transparency, (str, Path)):
        if not os.path.exists(transparency):
            raise ValueError(f"Transparency mask file not found: {transparency}")
        return nib.load(transparency)
    raise TypeError(
        f"`transparency` must be BrainData, Nifti1Image, path, 'auto', or "
        f"None; got {type(transparency)}"
    )


_VALID_SURF_VIEWS = ("lateral", "medial", "dorsal", "ventral", "anterior", "posterior")
_VALID_SURF_HEMIS = ("left", "right")


def _normalize_surf_views(view):
    """Return an ordered list of view names for plot_surf."""
    if view == "montage":
        return ["lateral", "medial"]
    views = [view] if isinstance(view, str) else list(view)
    bad = [v for v in views if v not in _VALID_SURF_VIEWS]
    if not views or bad:
        raise ValueError(
            f"Invalid view={view!r}. Each entry must be one of "
            f"{list(_VALID_SURF_VIEWS)}; got unknown entries {bad}."
        )
    return views


def _normalize_surf_hemis(hemi):
    """Return an ordered list of hemisphere names for plot_surf."""
    if hemi == "both":
        return ["left", "right"]
    hemis = [hemi] if isinstance(hemi, str) else list(hemi)
    bad = [h for h in hemis if h not in _VALID_SURF_HEMIS]
    if not hemis or bad:
        raise ValueError(
            f"Invalid hemi={hemi!r}. Must be 'left', 'right', 'both', or a "
            f"list thereof; got unknown entries {bad}."
        )
    return hemis


def plot_surf(
    brain,
    *,
    hemi="both",
    view="montage",
    surface="pial",
    template="fsaverage5",
    threshold=None,
    cmap=None,
    vmin=None,
    vmax=None,
    transparency="auto",
    bg_on_data=False,
    colorbar=True,
    colorbar_orientation="horizontal",
    figsize=(10, 8),
    title=None,
    radius=3.0,
    interpolation="linear",
    zoom=1.2,
    axes=None,
    save=None,
):
    """Plot volumetric data on fsaverage surfaces in a tight montage.

    Like nilearn's `plot_img_on_surf` but with tight framing (via
    `Axes3D.set_box_aspect(zoom=...)` + `set_axis_off`), an auto-applied
    transparency mask (same convention as `plot_flatmap`), and a single shared
    colorbar instead of one per subplot.

    The grid is `len(view) × len(hemi)` — rows are views, columns are hemispheres.

    Args:
        brain (BrainData | nibabel.Nifti1Image | str | Path): MNI-space image to
            plot.
        hemi (str | list): `'left'`, `'right'`, `'both'` (default), or a list
            subset like `['left']`.
        view (str | list): `'montage'` (default, → `['lateral', 'medial']`), a
            single view string, or any list subset of `'lateral'`, `'medial'`,
            `'dorsal'`, `'ventral'`, `'anterior'`, `'posterior'`.
        surface (str): fsaverage mesh to render on. One of `'pial'` (default),
            `'inflated'`, `'white'`, `'sphere'`.
        template (str): fsaverage resolution (`'fsaverage3'` … `'fsaverage'`).
            Default `'fsaverage5'`.
        threshold (float | str, optional): Absolute cutoff (`0.3`) or percentile
            string (`'95%'`).
        cmap (str, optional): Matplotlib colormap. By default, positive-only
            maps use ``"Reds"``, negative-only maps use ``"Blues_r"``, and
            mixed maps use ``"RdBu_r"``.
        vmin (float, optional): Colormap lower bound. Defaults to zero for
            positive-only maps, the data minimum for negative-only maps, and
            negative max-absolute value for mixed maps.
        vmax (float, optional): Colormap upper bound. Defaults to the data
            maximum for positive-only maps, zero for negative-only maps, and
            max-absolute value for mixed maps.
        transparency (BrainData | nibabel.Nifti1Image | str | Path | None): Binary
            mask used to NaN-out vertices outside the mask so the background shines
            through. `'auto'` (default) uses `BrainData.mask`; None disables masking.
        bg_on_data (bool): Whether to multiply data by the background. Default False.
        colorbar (bool): Show a single shared colorbar. Default True.
        colorbar_orientation (str): `'horizontal'` (default) or `'vertical'`.
        figsize (tuple): Figure size. Default (10, 8).
        title (str, optional): Figure title.
        radius (float): `vol_to_surf` sampling radius. Default 3.0.
        interpolation (str): `vol_to_surf` interpolation. Default `'linear'`.
        zoom (float): Zoom factor for each 3-D axis (`Axes3D.set_box_aspect`).
            Default 1.2; try 1.4 for the tightest clean framing.
        axes (np.ndarray, optional): Pre-existing `Axes3D` array to draw into, of
            shape `(len(view), len(hemi))`.
        save (str, optional): Path to save the figure.

    Returns:
        matplotlib.figure.Figure: The surface figure.
    """
    from nilearn import datasets
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from nltools.data import BrainData

    # --- validate input up front (before any network or surface work) ----
    if isinstance(brain, BrainData) and brain.is_empty:
        raise ValueError("Cannot plot empty BrainData object")

    views = _normalize_surf_views(view)
    hemis = _normalize_surf_hemis(hemi)

    # fsaverage stores inflated surfaces under the "infl_*" key; translate
    # the more readable public name to the internal key.
    surf_key_map = {
        "pial": "pial",
        "inflated": "infl",
        "white": "white",
        "sphere": "sphere",
    }
    if surface not in surf_key_map:
        raise ValueError(
            f"Invalid surface={surface!r}. Must be one of {list(surf_key_map)}."
        )
    surf_key = surf_key_map[surface]

    # Resolve transparency *before* converting brain to nifti (we need access
    # to BrainData.mask for the "auto" default).
    mask_img = _resolve_transparency(transparency, brain)
    nifti_img = _resolve_brain_input(brain)

    # --- fetch surfaces and project --------------------------------------
    fs = datasets.fetch_surf_fsaverage(template)

    textures = {}
    for h in hemis:
        tex = vol_to_surf(
            nifti_img,
            fs[f"{surf_key}_{h}"],
            radius=radius,
            interpolation=interpolation,
        )
        if mask_img is not None:
            mk = vol_to_surf(
                mask_img,
                fs[f"{surf_key}_{h}"],
                radius=radius,
                interpolation="linear",
            )
            tex = np.where(mk >= 0.5, tex, np.nan)
        textures[h] = tex

    # Percentile threshold (computed across all vertices/hemis)
    if isinstance(threshold, str) and threshold.endswith("%"):
        pct = float(threshold[:-1])
        all_vals = np.concatenate([textures[h] for h in hemis])
        all_vals = all_vals[np.isfinite(all_vals)]
        threshold = (
            float(np.percentile(np.abs(all_vals), pct)) if len(all_vals) else None
        )

    all_vals = np.concatenate([textures[h] for h in hemis])
    range_vals = (
        all_vals[np.abs(all_vals) >= threshold] if threshold is not None else all_vals
    )
    cmap, vmin, vmax = _resolve_stat_map_defaults(
        range_vals, cmap=cmap, vmin=vmin, vmax=vmax
    )

    # --- figure / axes grid ----------------------------------------------
    nrows, ncols = len(views), len(hemis)
    if axes is None:
        fig, axes_arr = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            subplot_kw={"projection": "3d"},
            constrained_layout=True,
            squeeze=False,
        )
        owns_fig = True
    else:
        axes_arr = np.asarray(axes).reshape(nrows, ncols)
        fig = axes_arr.flat[0].figure
        owns_fig = False

    # --- draw each subplot -----------------------------------------------
    for r, v in enumerate(views):
        for c, h in enumerate(hemis):
            ax = axes_arr[r, c]
            plot_surf_stat_map(
                fs[f"{surf_key}_{h}"],
                textures[h],
                hemi=h,
                view=v,
                bg_map=fs[f"curv_{h}"],
                bg_on_data=bg_on_data,
                colorbar=False,  # shared colorbar below
                cmap=cmap,
                threshold=threshold,
                vmax=vmax,
                vmin=vmin,
                axes=ax,
                engine="matplotlib",
            )
            ax.set_box_aspect((1, 1, 1), zoom=zoom)
            ax.set_axis_off()

    # --- shared colorbar --------------------------------------------------
    if colorbar:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=axes_arr.ravel().tolist(),
            orientation=colorbar_orientation,
            fraction=0.03,
            pad=0.02,
            shrink=0.7,
        )

    if title is not None:
        fig.suptitle(title, fontsize=14)

    if save is not None:
        fig.savefig(save, bbox_inches="tight", facecolor="white", dpi=300)

    if owns_fig:
        plt.close(fig)
    return fig


def plot_flatmap(
    brain,
    *,
    threshold=None,
    cmap=None,
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
        brain (BrainData | nibabel.Nifti1Image | str | Path): Image to plot. Data
            must be in MNI152 space.
        threshold (float or str, optional): Values below this absolute
            threshold are masked. Can be a float or percentile string
            like '95%'. Defaults to None (no threshold).
        cmap (str, optional): Matplotlib colormap. The default is ``"Reds"``
            for positive-only maps, ``"Blues_r"`` for negative-only maps, and
            ``"RdBu_r"`` for mixed maps.
        vmax (float, optional): Maximum value. Defaults to the positive data
            maximum, zero for negative-only data, or max-absolute value for
            mixed data.
        vmin (float, optional): Minimum value. Defaults to zero for positive-only
            data, the negative data minimum, or negative max-absolute value for
            mixed data.
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
        transparency (BrainData | nibabel.Nifti1Image | str | Path | None):
            Binary mask used to render vertices outside the mask as
            transparent (so the curvature shows through). `'auto'` (default)
            uses the input `BrainData`'s `.mask` when available, matching
            the behavior of the volumetric `.plot()`. Pass None to
            disable masking entirely.
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

        ```python
        from nltools.plotting import plot_flatmap
        from nltools.data import BrainData

        brain = BrainData("stats.nii.gz")
        fig = plot_flatmap(brain)
        ```

        Thresholded with custom colormap:

        ```python
        fig = plot_flatmap(brain, threshold=2.5, cmap="hot")
        ```

        Percentile threshold, no curvature:

        ```python
        fig = plot_flatmap(brain, threshold="95%", with_curvature=False)
        ```

        High resolution for publication:

        ```python
        fig = plot_flatmap(brain, template="fsaverage6", figsize=(16, 8))
        fig.savefig("flatmap.pdf", dpi=300)
        ```

    Note:
        Data is projected from MNI152 space to fsaverage surface space, so small
        alignment differences are expected at boundaries. Higher resolution
        templates (fsaverage6, fsaverage) produce sharper images but take longer
        to render. The flat surfaces are cached by nilearn after the first
        download (~50MB for fsaverage5).
    """
    from nilearn import datasets, surface
    import nibabel as nib
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    # Resolve transparency mask *before* converting input to nifti (we need
    # access to BrainData.mask for the "auto" default).
    mask_img = _resolve_transparency(transparency, brain)

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

    # Project the transparency mask to the surface and NaN-out vertices
    # outside the mask so the curvature shows through cleanly.
    if mask_img is not None:
        mask_left = surface.vol_to_surf(
            mask_img, fs["pial_left"], radius=radius, interpolation="linear"
        )
        mask_right = surface.vol_to_surf(
            mask_img, fs["pial_right"], radius=radius, interpolation="linear"
        )
        texture_left = np.where(mask_left >= 0.5, texture_left, np.nan)
        texture_right = np.where(mask_right >= 0.5, texture_right, np.nan)

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

    all_values = np.concatenate([texture_left, texture_right])
    range_values = (
        all_values[np.abs(all_values) >= threshold]
        if threshold is not None
        else all_values
    )
    cmap, vmin, vmax = _resolve_stat_map_defaults(
        range_values, cmap=cmap, vmin=vmin, vmax=vmax
    )

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

    # Create figure and axes. Track ownership so caller-supplied axes keep
    # their figure on pyplot's tracker.
    if axes is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        owns_fig = True
    else:
        ax = axes
        fig = ax.figure
        owns_fig = False

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

    if owns_fig:
        plt.close(fig)
    return fig
