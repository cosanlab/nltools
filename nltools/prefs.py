import os
import re
import warnings
from dataclasses import dataclass, field
from os.path import dirname, join
from typing import Literal

import numpy as np

__all__ = ["MNI_Template", "resolve_template_name"]

# Maps template names to version codes used in filenames
_VERSION_MAP = {
    "default": "fsl",
    "nilearn": "a",
    "fmriprep": "c",
}

# Reverse mapping: version code → template name
_VERSION_TO_TEMPLATE = {v: k for k, v in _VERSION_MAP.items()}

# Which resolutions each template supports
_SUPPORTED_RESOLUTIONS = {
    "default": [2, 3],
    "nilearn": [1, 2, 3],
    "fmriprep": [1, 2],
}

# Template search priority (used by match_resolution)
_TEMPLATE_PRIORITY = ["default", "nilearn", "fmriprep"]


def _resolve_paths(template: str, resolution: int) -> dict[str, str]:
    """Build mask/brain/plot paths for a template + resolution.

    Single source of truth for path construction and old/new naming fallback.

    Args:
        template: Template name ('default', 'nilearn', 'fmriprep').
        resolution: Resolution in mm.

    Returns:
        Dict with 'mask', 'brain', 'plot' paths.

    Raises:
        ValueError: If template or resolution is invalid.
        FileNotFoundError: If any template file is missing.
    """
    if template not in _VERSION_MAP:
        raise ValueError(f"Unknown template: {template}")
    if resolution not in _SUPPORTED_RESOLUTIONS.get(template, []):
        raise ValueError(
            f"Resolution {resolution}mm not supported for '{template}'. "
            f"Supported: {_SUPPORTED_RESOLUTIONS[template]}"
        )

    version = _VERSION_MAP[template]
    base_path = join(dirname(__file__), "resources", "niftis", template)
    res_str = f"{resolution}mm"

    paths = {}
    for file_type, attr in [("mask", "mask"), ("brain", "brain"), ("T1", "plot")]:
        new = join(base_path, f"{res_str}-MNI152-2009{version}-{file_type}.nii.gz")
        old = join(base_path, f"MNI152_{res_str}_{file_type}.nii.gz")

        if not os.path.exists(new) and os.path.exists(old):
            paths[attr] = old
        elif os.path.exists(new):
            paths[attr] = new
        else:
            raise FileNotFoundError(
                f"Template file not found: {new}\n"
                f"This suggests an incomplete installation or missing template files."
            )

    return paths


@dataclass
class MNI_Template_Factory:
    """Global MNI template configuration.

    This class manages the global MNI template settings used throughout nltools.
    Users should interact with the exported ``MNI_Template`` instance rather than
    creating new instances.

    Args:
        template ({'default', 'nilearn', 'fmriprep'}): Template variant to use.
        resolution ({1, 2, 3}): Resolution in mm. Not all resolutions are
            available for all templates.

    Attributes:
        mask (str): Path to the brain mask file
        brain (str): Path to the brain-extracted image
        plot (str): Path to the full T1 image for plotting

    Examples:
        >>> from nltools.prefs import MNI_Template
        >>> MNI_Template.template = 'fmriprep'
        >>> MNI_Template.resolution = 1
        >>> print(MNI_Template.mask)
    """

    template: Literal["default", "nilearn", "fmriprep"] = "default"
    resolution: Literal[1, 2, 3] = 2

    # Auto-populated paths
    mask: str = field(init=False)
    brain: str = field(init=False)
    plot: str = field(init=False)

    # Expose as class attribute for external access
    _supported_combinations = _SUPPORTED_RESOLUTIONS

    def __post_init__(self):
        self._update_paths()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in ("template", "resolution") and hasattr(self, "_update_paths"):
            self._update_paths()

    def __repr__(self) -> str:
        return (
            f"MNI_Template(template='{self.template}', resolution={self.resolution}mm)\n"
            f"  mask: {os.path.basename(self.mask)}\n"
            f"  brain: {os.path.basename(self.brain)}\n"
            f"  plot: {os.path.basename(self.plot)}"
        )

    def _update_paths(self):
        """Validate current settings and resolve file paths."""
        paths = _resolve_paths(self.template, self.resolution)
        # Use object.__setattr__ to avoid re-triggering _update_paths
        for attr in ("mask", "brain", "plot"):
            object.__setattr__(self, attr, paths[attr])

    def resolve_paths(
        self, template: str | None = None, resolution: int | None = None
    ) -> dict[str, str]:
        """Build mask/brain/plot paths for a given template + resolution.

        Defaults to the current global settings if arguments are omitted.

        Args:
            template: Template name. Defaults to ``self.template``.
            resolution: Resolution in mm. Defaults to ``self.resolution``.

        Returns:
            Dict with keys ``'mask'``, ``'brain'``, ``'plot'`` mapping to
            file paths.
        """
        return _resolve_paths(
            template if template is not None else self.template,
            resolution if resolution is not None else self.resolution,
        )

    def match_resolution(
        self,
        affine: np.ndarray,
        prefer_exact: bool = True,
        warn_resample: bool = True,
    ) -> dict:
        """Find the best matching template for a given affine matrix.

        Searches all available templates by priority and returns the one whose
        resolution most closely matches the data.

        Args:
            affine: 4x4 affine matrix from a NIfTI image.
            prefer_exact: If True, prefer an exact resolution match.
            warn_resample: If True, emit a warning when the data resolution
                doesn't exactly match the selected template.

        Returns:
            Dict with keys: ``'template'``, ``'resolution'``,
            ``'mask_path'``, ``'brain_path'``, ``'plot_path'``,
            ``'match_distance'``.
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

        # Try exact match first
        best_template = None
        best_resolution = None
        best_distance = float("inf")

        if prefer_exact:
            for tmpl in _TEMPLATE_PRIORITY:
                if resolution in _SUPPORTED_RESOLUTIONS[tmpl]:
                    best_template, best_resolution, best_distance = tmpl, resolution, 0
                    break

        # Fall back to closest resolution
        if best_template is None:
            all_res = {r for res in _SUPPORTED_RESOLUTIONS.values() for r in res}
            closest = min(all_res, key=lambda x: abs(x - resolution))
            best_distance = abs(closest - resolution)

            for tmpl in _TEMPLATE_PRIORITY:
                if closest in _SUPPORTED_RESOLUTIONS[tmpl]:
                    best_template, best_resolution = tmpl, closest
                    break

            if best_distance > 0 and warn_resample:
                warnings.warn(
                    f"\nData resolution ({resolution_float:.3f}mm) doesn't exactly "
                    f"match template: {best_template} {best_resolution}mm.",
                    UserWarning,
                    stacklevel=3,
                )

        paths = _resolve_paths(best_template, best_resolution)
        return {
            "template": best_template,
            "resolution": best_resolution,
            "mask_path": paths["mask"],
            "brain_path": paths["brain"],
            "plot_path": paths["plot"],
            "match_distance": best_distance,
        }

    def get_bg_image(self, affine: np.ndarray, img_type: str = "brain") -> str:
        """Get a background image path matching a data resolution.

        Uses the *current* template setting (``self.template``) and finds
        the matching resolution from the affine. Used by plotting functions
        to pick an appropriate background anatomical.

        Args:
            affine: 4x4 affine matrix from a BrainData's masker.
            img_type: ``'brain'`` for brain-extracted image or
                ``'plot'`` for full T1.

        Returns:
            Path to the template image file.

        Raises:
            ValueError: If voxels are non-isotropic or resolution is
                unsupported.
        """
        if img_type not in ("brain", "plot"):
            raise ValueError("img_type must be 'brain' or 'plot'")

        res_array = np.abs(np.diag(affine[:3, :3]))
        voxel_dims = np.unique(res_array)
        if len(voxel_dims) != 1:
            raise ValueError(
                "Voxels are not isotropic and cannot be visualized in standard space"
            )

        resolution = int(voxel_dims[0])

        if resolution not in _SUPPORTED_RESOLUTIONS.get(self.template, []):
            # Fall back to current template's default
            return getattr(self, img_type)

        paths = _resolve_paths(self.template, resolution)
        return paths[img_type]


def resolve_template_name(template_name: str, file_type: str = "mask") -> str:
    """Resolve a template name string to a file path.

    Supports template names in the format ``'{res}mm-MNI152-2009{version}'``.

    Args:
        template_name: e.g. ``'2mm-MNI152-2009c'``, ``'3mm-MNI152-2009a'``.
        file_type: ``'mask'``, ``'brain'``, or ``'T1'``. Default: ``'mask'``.

    Returns:
        Full path to the template file.

    Examples:
        >>> resolve_template_name('2mm-MNI152-2009c')
        '/path/to/nltools/resources/niftis/fmriprep/2mm-MNI152-2009c-mask.nii.gz'
    """
    if file_type not in ("mask", "brain", "T1"):
        raise ValueError(
            f"file_type must be 'mask', 'brain', or 'T1'. Got: {file_type}"
        )

    match = re.match(r"^(\d+)mm-MNI152-2009([acfsl]+)$", template_name)
    if not match:
        raise ValueError(
            f"Invalid template name format: '{template_name}'. "
            f"Expected: '{{res}}mm-MNI152-2009{{version}}' "
            f"(e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')"
        )

    resolution = int(match.group(1))
    version_code = match.group(2)

    if version_code not in _VERSION_TO_TEMPLATE:
        raise ValueError(
            f"Unknown version code '{version_code}' in '{template_name}'. "
            f"Supported: 'fsl' (default), 'a' (nilearn), 'c' (fmriprep)"
        )

    template = _VERSION_TO_TEMPLATE[version_code]
    # file_type uses 'T1' for plot, but resolve_paths returns 'plot' key
    attr = "plot" if file_type == "T1" else file_type
    return _resolve_paths(template, resolution)[attr]


# Singleton instance — users interact with this, not the class directly.
MNI_Template = MNI_Template_Factory()
