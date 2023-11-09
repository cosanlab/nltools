import os

__all__ = ["MNI_Template"]


class MNI_Template_Factory(object):
    """Class to build the default MNI_Template instance. This should never be used
    directly, instead just `from nltools.prefs import MNI_Template` and update that
    object's attributes to change MNI templates."""

    def __init__(
        self,
        resolution="2mm",
        mask_type="with_ventricles",
        mni_version="nonlin_6thgen",
    ):
        self._supported_resolutions = ["2mm", "3mm"]
        self._supported_mni_versions = ["nonlin_6thgen", "2009a", "2009c"]
        # Only applies to nonlin_6thgen
        self._supported_mask_types = ["with_ventricles", "no_ventricles"]

        self._resolution = resolution
        self._mask_type = mask_type
        self._mni_version = mni_version

        # Auto-populated (derive) mask, brain, plot
        # This also always called on attribute access so the latest paths
        # are resolved and can be safely used like nib.load(MNI_Template.mask)
        # after updating an attribute, e.g. MNI_Template.resolution = 3
        # Avoids having to do nib.load(resolve_mni_path(MNI_Template).mask) like
        # we were previously
        self.resolve_paths()

    def __repr__(self) -> str:
        if self.mni_version == "nonlin_6thgen":
            return f"Current global template:\nresolution={self.resolution}\nmni_version={self.mni_version}\nmask_type={self.mask_type}\nmask={self.mask}\nbrain={self.brain}\nplot={self.plot}"
        else:
            return f"Current global template:\nresolution={self.resolution}\nmni_version={self.mni_version}\nmask={self.mask}\nbrain={self.brain}\nplot={self.plot}"

    @property
    def mask(self):
        self.resolve_paths()
        return self._mask

    @property
    def plot(self):
        self.resolve_paths()
        return self._plot

    @property
    def brain(self):
        self.resolve_paths()
        return self._brain

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        if isinstance(resolution, (int, float)):
            resolution = f"{int(resolution)}mm"
        if resolution not in self._supported_resolutions:
            raise NotImplementedError(
                f"Nltools currently supports the following MNI template resolutions: {self._supported_resolutions}"
            )
        self._resolution = resolution
        self.resolve_paths()

    @property
    def mask_type(self):
        return self._mask_type

    @mask_type.setter
    def mask_type(self, mask_type):
        if mask_type not in self._supported_mask_types:
            raise NotImplementedError(
                f"Nltools currently supports the following MNI mask_types (only applies to nonlin_6thgen): {self._supported_mask_types}"
            )
        self._mask_type = mask_type
        self.resolve_paths()

    @property
    def mni_version(self):
        return self._mni_version

    @mni_version.setter
    def mni_version(self, mni_version):
        if mni_version not in self._supported_mni_versions:
            raise ValueError(
                f"Nltools currently supports the following MNI template versions: {self._supported_mni_versions}"
            )
        self._mni_version = mni_version
        self.resolve_paths()

    def resolve_paths(self):
        base_path = os.path.join(os.path.dirname(__file__), "resources") + os.path.sep
        if self._mni_version == "nonlin_6thgen":
            if self._resolution == "3mm":
                if self._mask_type == "with_ventricles":
                    self._mask = os.path.join(
                        base_path, "MNI152_T1_3mm_brain_mask.nii.gz"
                    )
                elif self._mask_type == "no_ventricles":
                    self._mask = os.path.join(
                        base_path,
                        "MNI152_T1_3mm_brain_mask_no_ventricles.nii.gz",
                    )
                self._plot = os.path.join(base_path, "MNI152_T1_3mm.nii.gz")
                self._brain = os.path.join(base_path, "MNI152_T1_3mm_brain.nii.gz")
            elif self._resolution == "2mm":
                if self._mask_type == "with_ventricles":
                    self._mask = os.path.join(
                        base_path, "MNI152_T1_2mm_brain_mask.nii.gz"
                    )
                elif self._mask_type == "no_ventricles":
                    self._mask = os.path.join(
                        base_path,
                        "MNI152_T1_2mm_brain_mask_no_ventricles.nii.gz",
                    )
                self._plot = os.path.join(base_path, "MNI152_T1_2mm.nii.gz")
                self._brain = os.path.join(base_path, "MNI152_T1_2mm_brain.nii.gz")
        elif self._mni_version == "2009c":
            pass
        elif self._mni_version == "2009a":
            pass


#  NOTE: We export this from the module and expect users to interact with it instead of
#  the class constructor above
MNI_Template = MNI_Template_Factory()
