from nltools.mask import create_sphere, roi_to_brain
import numpy as np
import pandas as pd

def test_create_sphere():
    # Test values update to reflect the fact that standard Brain_Data mask has few voxels because ventricles are 0'd out

    a = create_sphere(radius=10, coordinates=[0, 0, 0])
    assert np.sum(a.get_data()) >= 497  # 515
    a = create_sphere(radius=[10, 5], coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_data()) >= 553  # 571
    a = create_sphere(radius=10, coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_data()) >= 1013  # 1051

def test_append(sim_brain_data):
    assert sim_brain_data.append(sim_brain_data).shape()[0] == shape_2d[0]*2

def test_roi_to_brain():
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)
    s3 = create_sphere([0, -15, -8], radius=10)
    masks = Brain_Data([s1,s2,s3])

    d = [1,2,3]
    m = roi_to_brain(d, masks)
    assert np.all([np.any(m.data==x) for x in d])

    d = pd.Series([1.1, 2.1, 3.1])
    m = roi_to_brain(d, masks)
    assert np.all([np.any(m.data==x) for x in d])

    d = np.array([1, 2, 3])
    m = roi_to_brain(d, masks)
    assert np.all([np.any(m.data==x) for x in d])

    d = pd.DataFrame([np.ones(10)*x for x in [1, 2, 3]])
    m = roi_to_brain(d, masks)
    assert len(m) == d.shape[1]
    assert np.all([np.any(m[0].data==x) for x in d[0]])

    d = np.array([np.ones(10)*x for x in [1, 2, 3]])
    m = roi_to_brain(d, masks)
    assert len(m) == d.shape[1]
    assert np.all([np.any(m[0].data==x) for x in d[0]])