import warnings

from nltools.mask import (
    collapse_mask,
    create_sphere,
    expand_mask,
    roi_to_brain,
    roi_to_brain_from_atlas,
)
import nibabel as nib
from nltools.data import BrainData
import numpy as np
import pandas as pd
import pytest


def test_create_sphere():
    # Voxel counts on the standard BrainData mask, which has few voxels because
    # ventricles are 0'd out. Pinned with 5% quantization slack rather than a
    # lower bound: the pre-nilearn implementation resampled with continuous
    # interpolation and inflated these to 528 / 597 / 1056, which a `>=` bound
    # would not catch.
    a = create_sphere(radius=10, coordinates=[0, 0, 0])
    assert np.sum(a.get_fdata()) == pytest.approx(515, rel=0.05)
    a = create_sphere(radius=[10, 5], coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_fdata()) == pytest.approx(571, rel=0.05)
    a = create_sphere(radius=10, coordinates=[[0, 0, 0], [15, 0, 25]])
    assert np.sum(a.get_fdata()) == pytest.approx(1051, rel=0.05)


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


def test_roi_to_brain_drops_fitted_mask_state():
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)
    s3 = create_sphere([0, -15, -8], radius=10)
    masks = BrainData([s1, s2, s3])
    X = np.arange(3, dtype=float).reshape(-1, 1)
    masks.fit(model="ridge", X=X, ridge_alpha=1.0)

    result = roi_to_brain(np.ones((3, 2)), masks)

    assert result.model is None
    assert not hasattr(result, "X_")


def _isotropic_grid_mask(voxel_size, extent_mm=72.0):
    """An all-ones cubic mask with isotropic `voxel_size` mm voxels centered on the origin."""
    n = int(round(extent_mm / voxel_size))
    shape = (n, n, n)
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    affine[:3, 3] = -voxel_size * (np.array(shape) // 2)
    return nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)


@pytest.mark.parametrize("voxel_size", [2.0])
@pytest.mark.parametrize("radius", [6.0])
def test_create_sphere_radius_is_millimeters_on_any_grid(voxel_size, radius):
    """The radius is millimeters, so the sphere's volume is resolution-independent."""
    mask = _isotropic_grid_mask(voxel_size)

    sphere = create_sphere([0, 0, 0], radius=radius, mask=mask)

    n_voxels = int(np.sum(np.asarray(sphere.dataobj) > 0))
    volume = n_voxels * voxel_size**3
    analytic = 4.0 / 3.0 * np.pi * radius**3
    # Voxel quantization is the only source of error; it grows with voxel_size / radius.
    assert abs(volume - analytic) / analytic < 0.15


@pytest.mark.parametrize("voxel_size", [2.0])
def test_create_sphere_center_is_a_world_coordinate(voxel_size):
    """Centers are world (MNI) millimeters, resolved through the mask affine."""
    mask = _isotropic_grid_mask(voxel_size)
    coordinate = [12.0, -10.0, 8.0]

    sphere = create_sphere(coordinate, radius=9.0, mask=mask)

    data = np.asarray(sphere.dataobj)
    center_ijk = np.round(
        nib.affines.apply_affine(np.linalg.inv(mask.affine), coordinate)
    ).astype(int)
    assert data[tuple(center_ijk)] > 0

    world = nib.affines.apply_affine(mask.affine, np.argwhere(data > 0))
    assert np.allclose(world.mean(axis=0), coordinate, atol=voxel_size)


def test_create_sphere_empty_sphere_raises():
    """A center whose sphere holds no in-mask voxel is an error, not a silent empty ROI."""
    mask = _isotropic_grid_mask(2.0)

    with pytest.raises(ValueError, match="outside the mask"):
        create_sphere([500.0, 500.0, 500.0], radius=4.0, mask=mask)


def test_create_sphere_accepts_a_non_binary_mask():
    """A non-binary mask with an in-mask center must draw, not report a bad center."""
    data = np.zeros((20, 20, 20), dtype=np.float32)
    data[5:15, 5:15, 5:15] = 2.0
    mask = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
    # The in-mask cube spans world 10-28 mm on every axis; pick its middle.
    center = [20.0, 20.0, 20.0]

    sphere = create_sphere(center, radius=6.0, mask=mask)

    assert int(np.sum(np.asarray(sphere.dataobj) > 0)) > 0


def test_create_sphere_carries_the_mask_header():
    """The returned image keeps the mask's header, so zooms and codes round-trip."""
    mask = _isotropic_grid_mask(3.0)

    sphere = create_sphere([0.0, 0.0, 0.0], radius=9.0, mask=mask)

    assert sphere.header.get_zooms()[:3] == mask.header.get_zooms()[:3]
    assert sphere.header["sform_code"] == mask.header["sform_code"]


def test_expand_and_collapse_mask_are_nifti_safe():
    # Nifti tooling (and nilearn's `new_img_like`) cannot carry 64-bit ints, so
    # labeled masks must round-trip to NIfTI without a downcast warning.
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)

    labeled = BrainData(s1)
    labeled.data = np.where(BrainData(s2).data > 0, 2.0, labeled.data)

    expanded = expand_mask(labeled)
    assert expanded.data.dtype == np.int32

    collapsed = collapse_mask(expanded)
    assert collapsed.data.dtype == np.int32

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expanded[0].to_nifti()
        collapsed.to_nifti()


def test_collapse_mask_keeps_its_own_labels_as_int32():
    # `auto_label=False` labels each region with the mask's own value. Both
    # branches promise integer labels, so both must be NIfTI-safe.
    s1 = create_sphere([15, 10, -8], radius=10)
    s2 = create_sphere([-15, 10, -8], radius=10)

    masks = BrainData([s1, s2])
    masks.data = masks.data * np.array([[3.0], [7.0]])

    collapsed = collapse_mask(masks, auto_label=False)
    assert collapsed.data.dtype == np.int32
    assert set(np.unique(collapsed.data)) == {0, 3, 7}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        collapsed.to_nifti()


def _four_voxel_mask():
    """A 4-voxel all-ones brain mask, so collapse/expand run outside template space."""
    return nib.Nifti1Image(np.ones((4, 1, 1), dtype=np.float32), np.eye(4))


def test_collapse_mask_stays_in_the_input_space():
    # E-04: collapsing used to rebuild every intermediate against a template
    # mask, so a custom-space input came back as an all-zero template vector
    # attached to the original mask, and `to_nifti()` raised.
    mask_img = _four_voxel_mask()
    masks = BrainData(
        np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]), mask=mask_img
    )

    collapsed = collapse_mask(masks)

    assert collapsed.mask.shape == (4, 1, 1)
    assert collapsed.data.tolist() == [1, 1, 2, 2]
    assert collapsed.to_nifti().shape == (4, 1, 1)


def test_collapse_mask_drops_every_overlap():
    # E-03: only voxels shared by *all* masks were dropped, so with 3+ masks a
    # pairwise overlap kept the sum of its labels (voxel 1 below came back as 3,
    # indistinguishable from mask 3's own label).
    mask_img = _four_voxel_mask()
    masks = BrainData(
        np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        mask=mask_img,
    )

    collapsed = collapse_mask(masks)

    assert collapsed.data.tolist() == [1, 0, 2, 3]


def test_expand_mask_leaves_the_caller_s_data_alone():
    # E-13: the rounding to int32 was written back onto the input object, so a
    # caller's labeled image was silently rounded and retyped under it.
    labels = BrainData(np.array([[0.0, 1.2, 2.0, 0.0]]), mask=_four_voxel_mask())
    before = labels.data.copy()

    expanded = expand_mask(labels)

    assert labels.data.dtype == before.dtype
    assert np.array_equal(labels.data, before)
    assert expanded.data.dtype == np.int32
    assert expanded.data.tolist() == [[0, 1, 0, 0], [0, 0, 1, 0]]


def test_roi_to_brain_from_atlas_rejects_a_zero_roi_label():
    # E-14: label 0 is background and receives `fill`, as the docstring says. An
    # explicit roi_labels=[0, ...] used to paint its value over the background
    # instead, contradicting both the docstring and the automatic path.
    mask_img = _four_voxel_mask()
    atlas = nib.Nifti1Image(
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32).reshape(4, 1, 1), np.eye(4)
    )

    with pytest.raises(ValueError, match="background"):
        roi_to_brain_from_atlas(
            np.array([99.0, 7.0]), atlas, mask_img, roi_labels=[0, 1], fill=-1.0
        )

    with pytest.raises(ValueError, match="background"):
        roi_to_brain_from_atlas(
            np.array([[99.0, 7.0], [98.0, 6.0]]),
            atlas,
            mask_img,
            roi_labels=[0, 1],
            fill=-1.0,
        )


def test_roi_to_brain_from_atlas_rounds_float_noise_labels():
    # 0.9999999 is label 1 and 1.9999999 is label 2; truncation made the first
    # background (painted with `fill`) and merged the second into label 1.
    mask_img = _four_voxel_mask()
    atlas = nib.Nifti1Image(
        np.array([0.0, 0.9999999, 2.0000001, 1.9999999]).reshape(4, 1, 1), np.eye(4)
    )
    out = roi_to_brain_from_atlas(np.array([10.0, 20.0]), atlas, mask_img, fill=-1.0)
    np.testing.assert_array_equal(
        np.asarray(out.data).ravel(), [-1.0, 10.0, 20.0, 20.0]
    )
