"""Pure path-resolution helpers for MNI template files."""

import os
import re
from os.path import dirname, join

from .registry import SUPPORTED_RESOLUTIONS, VERSION_MAP, VERSION_TO_TEMPLATE

_RESOURCES = join(dirname(dirname(__file__)), "resources", "niftis")


def resolve_paths(template: str, resolution: int) -> dict[str, str]:
    """Build mask/brain/plot paths for a template + resolution.

    Args:
        template: Template name (``'default'``, ``'nilearn'``, ``'fmriprep'``).
        resolution: Resolution in mm.

    Returns:
        Dict with keys ``'mask'``, ``'brain'``, ``'plot'``.

    Raises:
        ValueError: If template or resolution is invalid.
        FileNotFoundError: If any template file is missing on disk.
    """
    if template not in VERSION_MAP:
        raise ValueError(
            f"Unknown template: {template!r}. Supported: {sorted(VERSION_MAP)}"
        )
    if resolution not in SUPPORTED_RESOLUTIONS.get(template, []):
        raise ValueError(
            f"Resolution {resolution}mm not supported for {template!r}. "
            f"Supported: {SUPPORTED_RESOLUTIONS[template]}"
        )

    version = VERSION_MAP[template]
    base = join(_RESOURCES, template)
    res_str = f"{resolution}mm"

    out: dict[str, str] = {}
    for file_type, key in [("mask", "mask"), ("brain", "brain"), ("T1", "plot")]:
        new = join(base, f"{res_str}-MNI152-2009{version}-{file_type}.nii.gz")
        old = join(base, f"MNI152_{res_str}_{file_type}.nii.gz")
        if not os.path.exists(new) and os.path.exists(old):
            out[key] = old
        elif os.path.exists(new):
            out[key] = new
        else:
            raise FileNotFoundError(
                f"Template file not found: {new}\n"
                f"This suggests an incomplete installation or missing template files."
            )
    return out


def resolve_template_name(template_name: str, file_type: str = "mask") -> str:
    """Resolve a template name string to a file path.

    Supports names of the form ``'{res}mm-MNI152-2009{version}'``.

    Args:
        template_name: e.g. ``'2mm-MNI152-2009c'``, ``'3mm-MNI152-2009a'``.
        file_type: ``'mask'``, ``'brain'``, or ``'T1'``.

    Returns:
        Absolute path to the requested template file.
    """
    if file_type not in ("mask", "brain", "T1"):
        raise ValueError(
            f"file_type must be 'mask', 'brain', or 'T1'. Got: {file_type!r}"
        )

    match = re.match(r"^(\d+)mm-MNI152-2009([acfsl]+)$", template_name)
    if not match:
        raise ValueError(
            f"Invalid template name format: {template_name!r}. "
            f"Expected: '{{res}}mm-MNI152-2009{{version}}' "
            f"(e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')"
        )

    resolution = int(match.group(1))
    version_code = match.group(2)

    if version_code not in VERSION_TO_TEMPLATE:
        raise ValueError(
            f"Unknown version code {version_code!r} in {template_name!r}. "
            f"Supported: 'fsl' (default), 'a' (nilearn), 'c' (fmriprep)"
        )

    template = VERSION_TO_TEMPLATE[version_code]
    key = "plot" if file_type == "T1" else file_type
    return resolve_paths(template, resolution)[key]
