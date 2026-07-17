"""Affine-based template matching and background-image selection."""

import warnings
from dataclasses import dataclass

import numpy as np

from .config import BrainSpaceConfig, get_brainspace
from .paths import resolve_paths
from .registry import SUPPORTED_RESOLUTIONS, TEMPLATE_PRIORITY


@dataclass(frozen=True)
class TemplateMatch:
    """Result of matching a data affine to a template.

    Attributes:
        template: Best-matching template name.
        resolution: Best-matching resolution in mm.
        mask_path: Path to the matched mask file.
        brain_path: Path to the matched brain file.
        plot_path: Path to the matched T1/plot file.
        match_distance: Absolute difference in mm between detected data
            resolution and the selected template resolution (0 for exact).
    """

    template: str
    resolution: int
    mask_path: str
    brain_path: str
    plot_path: str
    match_distance: float


def match_resolution(
    affine: np.ndarray,
    prefer_exact: bool = True,
    warn_resample: bool = True,
) -> TemplateMatch:
    """Find the best matching template for a given affine matrix.

    Searches available templates by priority and returns the one whose
    resolution most closely matches the data's voxel size.

    Args:
        affine: 4x4 affine matrix from a NIfTI image.
        prefer_exact: If True, prefer an exact resolution match.
        warn_resample: If True, emit a warning when data resolution doesn't
            exactly match the selected template.

    Returns:
        A `TemplateMatch`.

    Raises:
        ValueError: If detected resolution is outside a reasonable range.
    """
    res_array = np.abs(np.diag(affine[:3, :3]))
    voxel_dims = np.unique(res_array)
    resolution_float = (
        float(voxel_dims[0]) if len(voxel_dims) == 1 else float(np.mean(res_array))
    )
    resolution = int(np.round(resolution_float))

    if resolution < 1 or resolution > 10:
        raise ValueError(
            f"Detected resolution ({resolution_float}mm) is outside "
            f"reasonable range (1-10mm). Data may not be in standard MNI space."
        )

    best_template: str | None = None
    best_resolution: int | None = None
    best_distance: float = float("inf")

    if prefer_exact:
        for tmpl in TEMPLATE_PRIORITY:
            if resolution in SUPPORTED_RESOLUTIONS[tmpl]:
                best_template, best_resolution, best_distance = tmpl, resolution, 0.0
                break

    if best_template is None:
        all_res = {r for res in SUPPORTED_RESOLUTIONS.values() for r in res}
        closest = min(all_res, key=lambda x: abs(x - resolution))
        best_distance = float(abs(closest - resolution))
        for tmpl in TEMPLATE_PRIORITY:
            if closest in SUPPORTED_RESOLUTIONS[tmpl]:
                best_template, best_resolution = tmpl, closest
                break
        if best_distance > 0 and warn_resample:
            warnings.warn(
                f"\nData resolution ({resolution_float:.3f}mm) doesn't exactly "
                f"match template: {best_template} {best_resolution}mm.",
                UserWarning,
                stacklevel=3,
            )

    assert best_template is not None and best_resolution is not None
    paths = resolve_paths(best_template, best_resolution)
    return TemplateMatch(
        template=best_template,
        resolution=best_resolution,
        mask_path=paths["mask"],
        brain_path=paths["brain"],
        plot_path=paths["plot"],
        match_distance=best_distance,
    )


def is_standard_space(
    affine: np.ndarray,
    *,
    config: BrainSpaceConfig | None = None,
) -> tuple[bool, str | None]:
    """Check whether an affine is compatible with our MNI templates.

    A "standard space" affine has isotropic voxels at one of the supported
    template resolutions (the union of ``SUPPORTED_RESOLUTIONS``). Plotting
    surfaces (glass brain, flatmap, surface montage) and template-driven
    background lookup all assume this — non-isotropic or off-grid data
    would render in misleading positions.

    Args:
        affine: 4x4 affine matrix from a NIfTI image (typically
            ``bd.mask.affine``).
        config: Optional explicit ``BrainSpaceConfig``; defaults to the
            current global brain space (only the supported resolution set
            is consulted).

    Returns:
        ``(True, None)`` if compatible; otherwise ``(False, reason)`` with
        ``reason`` a one-line human-readable explanation suitable for
        embedding in an error message.
    """
    del config  # accepted for symmetry with get_bg_image; not needed today
    res_array = np.abs(np.diag(affine[:3, :3]))
    voxel_dims = np.unique(np.round(res_array, 3))
    if len(voxel_dims) != 1:
        zooms = tuple(round(float(r), 2) for r in res_array)
        return False, f"voxels are non-isotropic (zooms={zooms} mm)"
    res = float(voxel_dims[0])
    res_int = int(round(res))
    if abs(res - res_int) > 1e-3:
        return False, (
            f"voxel size {res:.3f}mm is not an integer-mm template resolution"
        )
    all_supported = sorted({r for rs in SUPPORTED_RESOLUTIONS.values() for r in rs})
    if res_int not in all_supported:
        return False, (
            f"voxel size {res_int}mm is not a supported MNI template "
            f"resolution (supported: {all_supported}mm)"
        )
    return True, None


def get_bg_image(
    affine: np.ndarray,
    img_type: str = "brain",
    config: BrainSpaceConfig | None = None,
) -> str:
    """Get a background image path matching a data resolution.

    Uses ``config`` (or the current global brain space) and finds the
    matching resolution from the affine. Used by plotting functions to pick
    an appropriate background anatomical.

    Args:
        affine: 4x4 affine matrix from a BrainData's masker.
        img_type: ``'brain'`` for brain-extracted image or ``'plot'`` for
            full T1.
        config: Optional explicit config; defaults to current global.

    Returns:
        Path to the template image file.

    Raises:
        ValueError: If voxels are non-isotropic or ``img_type`` is invalid.
    """
    if img_type not in ("brain", "plot"):
        raise ValueError("img_type must be 'brain' or 'plot'")

    cfg = config if config is not None else get_brainspace()

    res_array = np.abs(np.diag(affine[:3, :3]))
    voxel_dims = np.unique(res_array)
    if len(voxel_dims) != 1:
        raise ValueError(
            "Voxels are not isotropic and cannot be visualized in standard space"
        )

    resolution = int(round(float(voxel_dims[0])))

    if resolution not in SUPPORTED_RESOLUTIONS.get(cfg.template, []):
        return getattr(cfg, img_type)

    paths = resolve_paths(cfg.template, resolution)
    return paths[img_type]
