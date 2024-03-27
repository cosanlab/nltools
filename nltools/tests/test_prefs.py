from nltools.prefs import MNI_Template
from nltools.data import Brain_Data
import matplotlib.pyplot as plt
import pytest


def test_change_mni_attribute():
    # Defaults
    brain = Brain_Data()
    assert brain.mask.affine[1, 1] == 2.0
    assert "2mm" in MNI_Template.brain

    # Change global -> 3mm
    MNI_Template.resolution = "3mm"
    # Now default is 3mm
    new_brain = Brain_Data()
    assert new_brain.mask.affine[1, 1] == 3.0
    assert "3mm" in MNI_Template.brain

    # switch back
    MNI_Template.resolution = 2.0  # floats are cool
    assert MNI_Template.resolution == "2mm"
    assert "2mm" in MNI_Template.brain

    # Back to 2mm
    newer_brain = Brain_Data()
    assert newer_brain.mask.affine[1, 1] == 2.0

    with pytest.raises(NotImplementedError):
        MNI_Template.resolution = 1


def test_pref_and_plotting(sim_brain_data):
    # Smoke tests to make sure updating templates doesn't cause plotting issues
    # plot methods always refer to the resolution of the Brain_Data
    # instance *itself*

    # Should have no effect as simulated data is in 2mm space
    MNI_Template.resolution = "3mm"
    sim_brain_data.plot()

    MNI_Template.resolution = "2mm"
    sim_brain_data.plot()

    # But they do refer to the currently loaded MNI_Template to get the mni_version
    # resolution
    # TODO: A a test for using a different mni_version (e.g. 2009c) via MNI_Template and making suring plotting still works
    plt.close("all")
