from nltools.mask import create_sphere, roi_to_brain
from nltools.data import BrainData
import numpy as np
import pandas as pd
import polars as pl
import pytest


def test_create_sphere():
    # Test values update to reflect the fact that standard BrainData mask has few voxels because ventricles are 0'd out

    a = create_sphere(radius=10, coordinates=[0, 0, 0])
    assert np.sum(a.get_fdata()) >= 497  # 515
    a = create_sphere(radius=[10, 5], coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_fdata()) >= 553  # 571
    a = create_sphere(radius=10, coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_fdata()) >= 1013  # 1051


def test_roi_to_brain():
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)
    s3 = create_sphere([0, -15, -8], radius=10)
    masks = BrainData([s1, s2, s3])

    d = [1, 2, 3]
    m = roi_to_brain(d, masks)
    assert np.all([np.any(m.data == x) for x in d])

    d = pd.Series([1.1, 2.1, 3.1])
    m = roi_to_brain(d, masks)
    assert np.all([np.any(m.data == x) for x in d])

    d = np.array([1, 2, 3])
    m = roi_to_brain(d, masks)
    assert np.all([np.any(m.data == x) for x in d])

    d = pd.DataFrame([np.ones(10) * x for x in [1, 2, 3]])
    m = roi_to_brain(d, masks)
    assert len(m) == d.shape[1]
    assert np.all([np.any(m[0].data == x) for x in d[0]])

    d = np.array([np.ones(10) * x for x in [1, 2, 3]])
    m = roi_to_brain(d, masks)
    assert len(m) == d.shape[1]
    assert np.all([np.any(m[0].data == x) for x in d[0]])


def test_roi_to_brain_polars_inputs():
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)
    s3 = create_sphere([0, -15, -8], radius=10)
    masks = BrainData([s1, s2, s3])

    d_series = pl.Series("roi", [1.5, 2.5, 3.5])
    m = roi_to_brain(d_series, masks)
    assert np.all([np.any(m.data == x) for x in d_series.to_list()])

    d_df = pl.DataFrame({f"col{i}": [1.0, 2.0, 3.0] for i in range(4)})
    m = roi_to_brain(d_df, masks)
    assert len(m) == d_df.shape[1]

    with pytest.raises(ValueError, match="Data must"):
        roi_to_brain("not a valid input", masks)
