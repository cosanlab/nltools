from nltools.prefs import MNI_Template
from nltools.data import BrainData
import pytest


def setup_function():
    """Reset MNI_Template to defaults before each test."""
    # Reset in a safe order to avoid validation errors
    object.__setattr__(MNI_Template, "template", "default")
    object.__setattr__(MNI_Template, "resolution", 2)
    # Re-resolve paths after direct attribute setting
    MNI_Template._validate_and_resolve()


def test_change_resolution():
    # Defaults
    assert MNI_Template.template == "default"
    assert MNI_Template.resolution == 2
    brain = BrainData()
    assert brain.mask.affine[1, 1] == 2.0
    assert "2mm" in MNI_Template.brain

    # Change resolution -> 3mm
    MNI_Template.resolution = 3
    new_brain = BrainData()
    assert new_brain.mask.affine[1, 1] == 3.0
    assert "3mm" in MNI_Template.brain

    # Switch back
    MNI_Template.resolution = 2
    assert MNI_Template.resolution == 2
    assert "2mm" in MNI_Template.brain

    # Test invalid resolution for default template
    with pytest.raises(ValueError, match="Resolution 1mm is not supported"):
        MNI_Template.resolution = 1


def test_change_template():
    # Test default template
    assert MNI_Template.template == "default"
    assert "default" in MNI_Template.mask

    # Switch to fmriprep
    MNI_Template.template = "fmriprep"
    assert MNI_Template.template == "fmriprep"
    assert "fmriprep" in MNI_Template.mask

    # Switch to nilearn
    MNI_Template.template = "nilearn"
    assert MNI_Template.template == "nilearn"
    assert "nilearn" in MNI_Template.mask

    # Reset
    MNI_Template.template = "default"


def test_template_resolution_combinations():
    # Test valid combinations
    valid_combos = [
        ("default", 2),
        ("default", 3),
        ("nilearn", 1),
        ("nilearn", 2),
        ("nilearn", 3),
        ("fmriprep", 1),
        ("fmriprep", 2),
    ]

    for template, resolution in valid_combos:
        # Reset to default first to avoid validation errors
        object.__setattr__(MNI_Template, "template", template)
        object.__setattr__(MNI_Template, "resolution", resolution)
        MNI_Template._validate_and_resolve()

        assert MNI_Template.template == template
        assert MNI_Template.resolution == resolution
        assert f"{resolution}mm" in MNI_Template.mask

    # Test invalid combinations
    invalid_combos = [
        ("default", 1),
        ("fmriprep", 3),
    ]

    for template, resolution in invalid_combos:
        MNI_Template.template = template
        with pytest.raises(ValueError):
            MNI_Template.resolution = resolution


def test_file_paths():
    # Test that all paths follow the expected pattern (new naming convention)
    MNI_Template.template = "default"
    MNI_Template.resolution = 2

    assert MNI_Template.mask.endswith("2mm-MNI152-2009fsl-mask.nii.gz")
    assert MNI_Template.brain.endswith("2mm-MNI152-2009fsl-brain.nii.gz")
    assert MNI_Template.plot.endswith("2mm-MNI152-2009fsl-T1.nii.gz")

    # Test different template (fmriprep uses 2009c variant)
    MNI_Template.template = "fmriprep"
    MNI_Template.resolution = 1

    assert MNI_Template.mask.endswith("1mm-MNI152-2009c-mask.nii.gz")
    assert MNI_Template.brain.endswith("1mm-MNI152-2009c-brain.nii.gz")
    assert MNI_Template.plot.endswith("1mm-MNI152-2009c-T1.nii.gz")


def test_repr():
    # Test string representation
    MNI_Template.template = "default"
    MNI_Template.resolution = 2
    repr_str = repr(MNI_Template)
    assert "template='default'" in repr_str
    assert "resolution=2mm" in repr_str
    assert "mask: 2mm-MNI152-2009fsl-mask.nii.gz" in repr_str
    assert "brain: 2mm-MNI152-2009fsl-brain.nii.gz" in repr_str
    assert "plot: 2mm-MNI152-2009fsl-T1.nii.gz" in repr_str
