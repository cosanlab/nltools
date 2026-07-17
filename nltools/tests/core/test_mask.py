from nltools.mask import create_sphere, expand_mask, roi_to_brain
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


def test_create_sphere_float_radius_multiple_coords():
    # F156: a float (or numpy) scalar radius with multiple coordinate triples
    # must be broadcast across coordinates, not left scalar (which crashed at
    # zip(radius, coordinates)).
    a = create_sphere(radius=10.0, coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_fdata()) > 0
    a = create_sphere(radius=np.int64(10), coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_fdata()) > 0


def test_expand_mask_non_contiguous_labels():
    # F149: expand_mask must iterate over the actual non-zero label values, not
    # over the indices of the unique-values array. A {0, 5, 10}-labeled atlas
    # should yield two non-empty binary masks, not two all-zero masks.
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)

    labeled = BrainData(s1)
    data = np.zeros(labeled.data.shape)
    data[BrainData(s1).data > 0] = 5
    data[BrainData(s2).data > 0] = 10
    labeled.data = data

    expanded = expand_mask(labeled)
    assert len(expanded) == 2
    assert np.any(expanded[0].data == 1)
    assert np.any(expanded[1].data == 1)


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


def test_roi_to_brain_2d_background_is_zero():
    # F151: the 2-D branch must initialize uncovered voxels to 0 (matching the
    # 1-D branch), not 1.
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)
    s3 = create_sphere([0, -15, -8], radius=10)
    masks = BrainData([s1, s2, s3])

    d = np.array([np.ones(10) * x for x in [1, 2, 3]])
    m = roi_to_brain(d, masks)

    uncovered = masks.data.sum(axis=0) == 0
    assert np.any(uncovered)  # spheres don't tile the whole mask
    assert np.all(m.data[:, uncovered] == 0)


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
