import os
from dataclasses import dataclass, field
from typing import Literal
from os.path import dirname, join

__all__ = ["MNI_Template"]


@dataclass
class MNI_Template_Factory:
    """Global MNI template configuration.
    
    This class manages the global MNI template settings used throughout nltools.
    Users should interact with the exported MNI_Template instance rather than
    creating new instances.
    
    Parameters
    ----------
    template : {'default', 'nilearn', 'fmriprep'}
        Template variant to use. Each template represents a different MNI space:
        - 'default': Original MNI152 6th generation templates
        - 'nilearn': Nilearn's MNI152 templates  
        - 'fmriprep': fMRIPrep's MNI152NLin2009cAsym templates
    resolution : {1, 2, 3}
        Resolution in mm. Not all resolutions are available for all templates:
        - 'default': 2mm, 3mm
        - 'nilearn': 1mm, 2mm, 3mm
        - 'fmriprep': 1mm, 2mm
        
    Attributes
    ----------
    mask : str
        Path to the brain mask file
    brain : str
        Path to the brain-extracted image
    plot : str
        Path to the full T1 image for plotting
        
    Examples
    --------
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
        "fmriprep": [1, 2]
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
        if name in ["template", "resolution"] and hasattr(self, "_validate_and_resolve"):
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
        
        # Build paths based on template and resolution
        base_path = join(dirname(__file__), "resources", "niftis", self.template)
        res_str = f"{self.resolution}mm"
        
        # Set paths following the naming convention
        self.mask = join(base_path, f"MNI152_{res_str}_mask.nii.gz")
        self.brain = join(base_path, f"MNI152_{res_str}_brain.nii.gz")
        self.plot = join(base_path, f"MNI152_{res_str}_T1.nii.gz")
        
        # Verify files exist
        for attr, path in [("mask", self.mask), ("brain", self.brain), ("plot", self.plot)]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Template file not found: {path}\n"
                    f"This suggests an incomplete installation or missing template files."
                )


# NOTE: We export this from the module and expect users to interact with it instead of
# the class constructor above
MNI_Template = MNI_Template_Factory()