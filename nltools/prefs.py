import os
from nltools.utils import get_resource_path

__all__ = ["MNI_Template", "resolve_mni_path"]


class MNI_Template_Factory(dict):
    """Class to build the default MNI_Template dictionary. This should never be used
    directly, instead just `from nltools.prefs import MNI_Template` and update that
    object's attributes to change MNI templates."""

    def __init__(
        self,
        resolution="2mm",
        mask_type="with_ventricles",
        mask=os.path.join(get_resource_path(), "MNI152_T1_2mm_brain_mask.nii.gz"),
        plot=os.path.join(get_resource_path(), "MNI152_T1_2mm.nii.gz"),
        brain=os.path.join(get_resource_path(), "MNI152_T1_2mm_brain.nii.gz"),
    ):
        self._resolution = resolution
        self._mask_type = mask_type
        self._mask = mask
        self._plot = plot
        self._brain = brain

        self.update(
            {
                "resolution": self.resolution,
                "mask_type": self.mask_type,
                "mask": self.mask,
                "plot": self.plot,
                "brain": self.brain,
            }
        )

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, resolution):
        if isinstance(resolution, (int, float)):
            resolution = f"{int(resolution)}mm"
        if resolution not in ["2mm", "3mm"]:
            raise NotImplementedError(
                "Only 2mm and 3mm resolutions are currently supported"
            )
        self._resolution = resolution
        self.update({"resolution": self._resolution})

    @property
    def mask_type(self):
        return self._mask_type

    @mask_type.setter
    def mask_type(self, mask_type):
        if mask_type not in ["with_ventricles", "no_ventricles"]:
            raise NotImplementedError(
                "Only 'with_ventricles' and 'no_ventricles' mask_types are currently supported"
            )
        self._mask_type = mask_type
        self.update({"mask_type": self._mask_type})

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask
        self.update({"mask": self._mask})

    @property
    def plot(self):
        return self._plot

    @plot.setter
    def plot(self, plot):
        self._plot = plot
        self.update({"plot": self._plot})

    @property
    def brain(self):
        return self._brain

    @brain.setter
    def brain(self, brain):
        self._brain = brain
        self.update({"brain": self._brain})


#  NOTE: We export this from the module and expect users to interact with it instead of
#  the class constructor above
MNI_Template = MNI_Template_Factory()


def resolve_mni_path(MNI_Template):
    """Helper function to resolve MNI path based on MNI_Template prefs setting."""

    res = MNI_Template["resolution"]
    m = MNI_Template["mask_type"]
    if not isinstance(res, str):
        raise ValueError("resolution must be provided as a string!")
    if not isinstance(m, str):
        raise ValueError("mask_type must be provided as a string!")

    if res == "3mm":
        if m == "with_ventricles":
            MNI_Template["mask"] = os.path.join(
                get_resource_path(), "MNI152_T1_3mm_brain_mask.nii.gz"
            )
        elif m == "no_ventricles":
            MNI_Template["mask"] = os.path.join(
                get_resource_path(), "MNI152_T1_3mm_brain_mask_no_ventricles.nii.gz"
            )
        else:
            raise ValueError(
                "Available mask_types are 'with_ventricles' or 'no_ventricles'"
            )

        MNI_Template["plot"] = os.path.join(get_resource_path(), "MNI152_T1_3mm.nii.gz")

        MNI_Template["brain"] = os.path.join(
            get_resource_path(), "MNI152_T1_3mm_brain.nii.gz"
        )

    elif res == "2mm":
        if m == "with_ventricles":
            MNI_Template["mask"] = os.path.join(
                get_resource_path(), "MNI152_T1_2mm_brain_mask.nii.gz"
            )
        elif m == "no_ventricles":
            MNI_Template["mask"] = os.path.join(
                get_resource_path(), "MNI152_T1_2mm_brain_mask_no_ventricles.nii.gz"
            )
        else:
            raise ValueError(
                "Available mask_types are 'with_ventricles' or 'no_ventricles'"
            )

        MNI_Template["plot"] = os.path.join(get_resource_path(), "MNI152_T1_2mm.nii.gz")

        MNI_Template["brain"] = os.path.join(
            get_resource_path(), "MNI152_T1_2mm_brain.nii.gz"
        )
    else:
        raise ValueError("Available templates are '2mm' or '3mm'")
    return MNI_Template
