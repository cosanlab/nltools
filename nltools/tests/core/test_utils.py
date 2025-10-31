from nltools.utils import check_brain_data, check_brain_data_is_single
from nltools.mask import create_sphere
from nltools.data import BrainData
import numpy as np


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
