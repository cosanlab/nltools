"""Static registry of supported MNI templates."""

from typing import Literal

TemplateName = Literal["default", "nilearn", "fmriprep"]
Resolution = Literal[1, 2, 3]

VERSION_MAP: dict[str, str] = {
    "default": "fsl",
    "nilearn": "a",
    "fmriprep": "c",
}

VERSION_TO_TEMPLATE: dict[str, str] = {v: k for k, v in VERSION_MAP.items()}

SUPPORTED_RESOLUTIONS: dict[str, list[int]] = {
    "default": [2, 3],
    "nilearn": [1, 2, 3],
    "fmriprep": [1, 2],
}

TEMPLATE_PRIORITY: list[str] = ["default", "nilearn", "fmriprep"]
