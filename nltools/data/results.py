from typing import Dict, Any
import nibabel as nib
from nltools.data import Brain_Data


class ResultsContainer(object):
    """A genericcontainer that dynamically creates attributes from a dictionary of string: Nifti1Image entries, where each attribute is a Brain_Data instance initialized from the corresponding Nifti image.

    Args:
        images_dict (Dict[str, nib.Nifti1Image]): Dictionary mapping attribute names to Nifti images.

    Example:
        >>> rc = ResultsContainer({'foo': img1, 'bar': img2})
        >>> rc.foo  # Brain_Data instance
        >>> rc.bar  # Brain_Data instance
    """

    def __init__(self, images=None):
        self._is_single = True
        self.data = []
        if isinstance(images, dict):
            for key, img in images.items():
                if not isinstance(img, nib.Nifti1Image):
                    raise TypeError(
                        f"Value for key '{key}' is not a Nifti1Image it's a {type(img)}."
                    )
                if key == "stat":
                    new_key = "t"
                elif key == "p_value":
                    new_key = "p"
                elif key == "effect_size":
                    new_key = "beta"
                elif key == "effect_variance":
                    new_key = "se"
                else:
                    new_key = key
                setattr(self, new_key, Brain_Data(img))
        elif isinstance(images, list):
            self._is_single = False
            for img in images:
                self.data.append(ResultsContainer(img))

    def __repr__(self):
        attr_names = [k for k in self.__dict__.keys()]
        return f"ResultsContainer(attributes={attr_names})"

    def append(self, image_dict):
        self._is_single = False
        # image_dict = (
        #     ResultsContainer(image_dict)
        #     if not isinstance(image_dict, ResultsContainer)
        #     else image_dict
        # )
        self.data.append(image_dict)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
