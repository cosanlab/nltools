from nltools.prefs import MNI_Template
from nltools.data import Brain_Data


def test_change_mni_resolution():
    assert MNI_Template["resolution"] == "2mm"
    brain = Brain_Data()
    assert brain.mask.affine[1, 1] == 2.0
    MNI_Template["resolution"] = "3mm"
    new_brain = Brain_Data()
    assert new_brain.mask.affine[1, 1] == 3.0
