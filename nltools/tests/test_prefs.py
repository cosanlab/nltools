from nltools.prefs import MNI_Template
from nltools.data import Brain_Data
import pytest


def test_change_mni_resolution():

    # Defaults
    brain = Brain_Data()
    assert brain.mask.affine[1, 1] == 2.0
    assert MNI_Template["resolution"] == "2mm"

    # -> 3mm
    MNI_Template["resolution"] = "3mm"
    new_brain = Brain_Data()
    assert new_brain.mask.affine[1, 1] == 3.0

    # switch back and test attribute setting
    MNI_Template.resolution = 2.0  # floats are cool
    assert MNI_Template["resolution"] == "2mm"

    newer_brain = Brain_Data()
    assert newer_brain.mask.affine[1, 1] == 2.0

    with pytest.raises(NotImplementedError):
        MNI_Template.resolution = 1
