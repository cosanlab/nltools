
from nltools.utils import check_brain_data, check_brain_data_is_single
from nltools.mask import create_sphere
import numpy as np

def check_brain_data(sim_brain_data):
    mask = Brain_Data(create_sphere([15, 10, -8], radius=10))
    a = check_brain_data(sim_brain_data)
    assert isinstance(a, Brain_Data)
    b = check_brain_data(sim_brain_data, mask=mask)
    assert isinstance(b, Brain_Data)
    assert b.shape()[1] == np.sum(mask.data==1)

def check_brain_data_is_single(sim_brain_data):
    assert not check_brain_data_is_single(sim_dat)
    assert check_brain_data_is_single(sim_dat[0])