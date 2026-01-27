import os
import re
from dataclasses import dataclass, field
from typing import Literal
from os.path import dirname, join

__all__ = ["MNI_Template", "resolve_template_name"]


@dataclass
class MNI_Template_Factory:
    """Global MNI template configuration.

    This class manages the global MNI template settings used throughout nltools.
    Users should interact with the exported MNI_Template instance rather than
    creating new instances.

    Args:
        template ({'default', 'nilearn', 'fmriprep'}): Template variant to use. Each
            template represents a different MNI space:
            - 'default': Original MNI152 6th generation templates
            - 'nilearn': Nilearn's MNI152 templates
            - 'fmriprep': fMRIPrep's MNI152NLin2009cAsym templates
        resolution ({1, 2, 3}): Resolution in mm. Not all resolutions are available
            for all templates:
            - 'default': 2mm, 3mm
            - 'nilearn': 1mm, 2mm, 3mm
            - 'fmriprep': 1mm, 2mm

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

    # Define supported combinations
    _supported_combinations = {
        "default": [2, 3],
        "nilearn": [1, 2, 3],
        "fmriprep": [1, 2],
    }

    def __post_init__(self):
        """Initialize paths after dataclass creation."""
        self._validate_and_resolve()

    def __setattr__(self, name, value):
        """Custom setter to re-resolve paths when attributes change."""
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, name, value)
        # Only resolve paths if we're setting template or resolution
        # and the object has been fully initialized
        if name in ["template", "resolution"] and hasattr(
            self, "_validate_and_resolve"
        ):
            self._validate_and_resolve()

    def __repr__(self) -> str:
        return (
            f"MNI_Template(template='{self.template}', resolution={self.resolution}mm)\n"
            f"  mask: {os.path.basename(self.mask)}\n"
            f"  brain: {os.path.basename(self.brain)}\n"
            f"  plot: {os.path.basename(self.plot)}"
        )

    def _validate_and_resolve(self):
        """Validate inputs and resolve file paths."""
        # Validate resolution is supported for this template
        if self.resolution not in self._supported_combinations.get(self.template, []):
            raise ValueError(
                f"Resolution {self.resolution}mm is not supported for template '{self.template}'. "
                f"Supported resolutions: {self._supported_combinations[self.template]}"
            )

        # Map template names to version codes
        version_map = {
            "default": "fsl",
            "nilearn": "a",
            "fmriprep": "c",
        }

        version = version_map.get(self.template)
        if version is None:
            raise ValueError(f"Unknown template: {self.template}")

        # Build paths based on template and resolution using new naming convention
        base_path = join(dirname(__file__), "resources", "niftis", self.template)
        res_str = f"{self.resolution}mm"

        # Set paths following the new naming convention: {res}mm-MNI152-2009{version}-{type}.nii.gz
        self.mask = join(base_path, f"{res_str}-MNI152-2009{version}-mask.nii.gz")
        self.brain = join(base_path, f"{res_str}-MNI152-2009{version}-brain.nii.gz")
        self.plot = join(base_path, f"{res_str}-MNI152-2009{version}-T1.nii.gz")

        # Fallback to old naming convention for backward compatibility
        old_mask = join(base_path, f"MNI152_{res_str}_mask.nii.gz")
        old_brain = join(base_path, f"MNI152_{res_str}_brain.nii.gz")
        old_plot = join(base_path, f"MNI152_{res_str}_T1.nii.gz")

        # Use new naming if available, otherwise fall back to old naming
        if not os.path.exists(self.mask) and os.path.exists(old_mask):
            self.mask = old_mask
        if not os.path.exists(self.brain) and os.path.exists(old_brain):
            self.brain = old_brain
        if not os.path.exists(self.plot) and os.path.exists(old_plot):
            self.plot = old_plot

        # Verify files exist
        for attr, path in [
            ("mask", self.mask),
            ("brain", self.brain),
            ("plot", self.plot),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Template file not found: {path}\n"
                    f"This suggests an incomplete installation or missing template files."
                )


def resolve_template_name(template_name: str, file_type: str = "mask") -> str:
    """
    Resolve a template name string to a file path.

    Supports template names in the format ``'{res}mm-MNI152-2009{version}'``
    where ``res`` is resolution (1, 2, or 3) and ``version`` is one of
    'fsl' (default/), 'a' (nilearn/), or 'c' (fmriprep/).

    Args:
        template_name: Template name string (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a')
        file_type: Type of file to return ('mask', 'brain', or 'T1'). Default: 'mask'

    Returns:
        str: Full path to the template file

    Raises:
        ValueError: If template_name format is invalid or file_type is unknown
        FileNotFoundError: If the template file doesn't exist

    Examples:
        >>> resolve_template_name('2mm-MNI152-2009c')
        '/path/to/nltools/resources/niftis/fmriprep/2mm-MNI152-2009c-mask.nii.gz'
        >>> resolve_template_name('3mm-MNI152-2009a', file_type='brain')
        '/path/to/nltools/resources/niftis/nilearn/3mm-MNI152-2009a-brain.nii.gz'
    """
    # Validate file_type
    if file_type not in ["mask", "brain", "T1"]:
        raise ValueError(
            f"file_type must be 'mask', 'brain', or 'T1'. Got: {file_type}"
        )

    # Parse template name using regex
    # Pattern: {res}mm-MNI152-2009{version}
    pattern = r"^(\d+)mm-MNI152-2009([acfsl]+)$"
    match = re.match(pattern, template_name)

    if not match:
        raise ValueError(
            f"Invalid template name format: '{template_name}'. "
            f"Expected format: '{{res}}mm-MNI152-2009{{version}}' "
            f"(e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl')"
        )

    resolution = int(match.group(1))
    version_code = match.group(2)

    # Map version codes to template directories
    version_to_template = {
        "fsl": "default",
        "a": "nilearn",
        "c": "fmriprep",
    }

    if version_code not in version_to_template:
        raise ValueError(
            f"Unknown version code '{version_code}' in template name '{template_name}'. "
            f"Supported versions: 'fsl' (default), 'a' (nilearn), 'c' (fmriprep)"
        )

    template_dir = version_to_template[version_code]

    # Validate resolution is supported for this template
    supported_combinations = {
        "default": [2, 3],
        "nilearn": [1, 2, 3],
        "fmriprep": [1, 2],
    }

    if resolution not in supported_combinations.get(template_dir, []):
        raise ValueError(
            f"Resolution {resolution}mm is not supported for template '{template_dir}'. "
            f"Supported resolutions: {supported_combinations[template_dir]}"
        )

    # Build file path
    base_path = join(dirname(__file__), "resources", "niftis", template_dir)
    file_path = join(
        base_path, f"{resolution}mm-MNI152-2009{version_code}-{file_type}.nii.gz"
    )

    # Fallback to old naming convention for backward compatibility
    old_file_path = join(base_path, f"MNI152_{resolution}mm_{file_type}.nii.gz")

    # Use new naming if available, otherwise fall back to old naming
    if not os.path.exists(file_path) and os.path.exists(old_file_path):
        file_path = old_file_path

    # Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Template file not found: {file_path}\n"
            f"This suggests an incomplete installation or missing template files."
        )

    return file_path


# NOTE: We export this from the module and expect users to interact with it instead of
# the class constructor above
MNI_Template = MNI_Template_Factory()
