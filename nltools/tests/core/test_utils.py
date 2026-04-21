from pathlib import Path

import numpy as np
import pytest

from nltools.data import BrainData
from nltools.data.braindata.utils import check_brain_data, check_brain_data_is_single
from nltools.mask import create_sphere


def test_check_brain_data(sim_brain_data):
    mask = BrainData(create_sphere([15, 10, -8], radius=10))
    a = check_brain_data(sim_brain_data)
    assert isinstance(a, BrainData)
    b = check_brain_data(sim_brain_data, mask=mask)
    assert isinstance(b, BrainData)
    assert b.shape[1] == np.sum(mask.data == 1)


def test_check_brain_data_is_single(sim_brain_data):
    assert not check_brain_data_is_single(sim_brain_data)
    assert check_brain_data_is_single(sim_brain_data[0])


def test_check_brain_data_from_nifti(sim_brain_data):
    """Nifti1Image input is coerced to BrainData (pre-existing behavior)."""
    nifti = sim_brain_data.to_nifti()
    out = check_brain_data(nifti)
    assert isinstance(out, BrainData)
    assert out.shape == sim_brain_data.shape


def test_check_brain_data_from_str_path(sim_brain_data, tmpdir):
    """File path (str) is now accepted — delegates to BrainData.__init__."""
    path = str(tmpdir.join("data.nii.gz"))
    sim_brain_data.to_nifti().to_filename(path)
    out = check_brain_data(path)
    assert isinstance(out, BrainData)
    assert out.shape == sim_brain_data.shape


def test_check_brain_data_from_pathlib(sim_brain_data, tmpdir):
    """Path objects also accepted."""
    path = Path(str(tmpdir.join("data.nii.gz")))
    sim_brain_data.to_nifti().to_filename(str(path))
    out = check_brain_data(path)
    assert isinstance(out, BrainData)
    assert out.shape == sim_brain_data.shape


def test_check_brain_data_from_list_of_paths(sim_brain_data, tmpdir):
    """List of file paths is now accepted."""
    p1 = str(tmpdir.join("a.nii.gz"))
    p2 = str(tmpdir.join("b.nii.gz"))
    sim_brain_data[0].to_nifti().to_filename(p1)
    sim_brain_data[1].to_nifti().to_filename(p2)
    out = check_brain_data([p1, p2])
    assert isinstance(out, BrainData)
    assert out.shape[0] == 2


def test_check_brain_data_rejects_unsupported_type():
    """Unsupported types raise TypeError from validate_data_type."""
    with pytest.raises(TypeError, match="Data must be"):
        check_brain_data(12345)
    with pytest.raises(TypeError, match="Data must be"):
        check_brain_data({"foo": "bar"})
