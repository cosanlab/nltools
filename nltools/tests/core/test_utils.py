import pytest

from nltools.data import BrainData
from nltools.data.braindata.utils import _check_brain_data


def test_check_brain_data_from_list_of_paths(sim_brain_data, tmpdir):
    """List of file paths is now accepted."""
    p1 = str(tmpdir.join("a.nii.gz"))
    p2 = str(tmpdir.join("b.nii.gz"))
    sim_brain_data[0].to_nifti().to_filename(p1)
    sim_brain_data[1].to_nifti().to_filename(p2)
    out = _check_brain_data([p1, p2])
    assert isinstance(out, BrainData)
    assert out.shape[0] == 2


def test_check_brain_data_rejects_unsupported_type():
    """Unsupported types raise TypeError from _validate_data_type."""
    with pytest.raises(TypeError, match="Data must be"):
        _check_brain_data(12345)
    with pytest.raises(TypeError, match="Data must be"):
        _check_brain_data({"foo": "bar"})
