"""`BrainData.align` wraps a value as `BrainData` only when it is spatial.

The v0.5.1 dict container is retained (`transformed`,
`transformation_matrix`, `common_model`, plus `disparity` and `scale` for
procrustes, plus `roi_labels` at ROI scale). A value is a `BrainData` when
its columns are a voxel axis matching the mask it carries; the SRM
`transformed` and `common_model` span the common model's feature axis, and
the `axis=1` transformation matrix spans images, so those stay raw arrays.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import polars as pl
import pytest

from nltools.data import BrainData

N_IMAGES = 30
N_VOXELS = 6
N_PARCELS = 2
N_REDUCED_FEATURES = 2

SRM_METHODS = ["deterministic_srm"]
SRM_KEYS = {"transformed", "transformation_matrix", "common_model"}
PROCRUSTES_KEYS = SRM_KEYS | {"disparity", "scale"}


def _brain_data(n_voxels, seed):
    """Build a small BrainData with `n_voxels` active voxels and row metadata."""
    rng = np.random.default_rng(seed)
    spatial_shape = (4, 3, 1)
    mask_values = np.zeros(spatial_shape, dtype=np.float32)
    mask_values.flat[:n_voxels] = 1.0
    values = rng.standard_normal((N_IMAGES, n_voxels))
    volume = np.zeros(spatial_shape + (N_IMAGES,))
    for image in range(N_IMAGES):
        volume.reshape(-1, N_IMAGES)[:n_voxels, image] = values[image]
    affine = np.eye(4)
    brain = BrainData(
        nib.Nifti1Image(volume, affine),
        mask=nib.Nifti1Image(mask_values, affine),
    )
    brain.X = pl.DataFrame(
        {"Intercept": np.ones(N_IMAGES), "X1": rng.standard_normal(N_IMAGES)}
    )
    brain.Y = pl.DataFrame({"Y": rng.standard_normal(N_IMAGES)})
    return brain


@pytest.fixture
def align_brain_data():
    """Source data with an evenly divisible voxel count."""
    return _brain_data(N_VOXELS, seed=0)


@pytest.fixture
def wider_target():
    """Procrustes target with more voxels than the source, forcing zero-padding."""
    return _brain_data(N_VOXELS + 2, seed=3)


@pytest.fixture
def align_atlas(align_brain_data):
    """Atlas splitting the mask into `N_PARCELS` equally sized parcels."""
    support = align_brain_data.mask.get_fdata() > 0
    labels = np.zeros(N_VOXELS, dtype=np.int16)
    per_parcel = N_VOXELS // N_PARCELS
    for parcel in range(N_PARCELS):
        labels[parcel * per_parcel : (parcel + 1) * per_parcel] = parcel + 1
    values = np.zeros(align_brain_data.mask.shape, dtype=np.int16)
    values[support] = labels
    return nib.Nifti1Image(values, align_brain_data.mask.affine)


@pytest.fixture
def whole_brain_model():
    """Common model with one feature per voxel of `align_brain_data`."""
    return np.random.default_rng(1).standard_normal((N_IMAGES, N_VOXELS))


@pytest.fixture
def reduced_model():
    """Common model with fewer features than `align_brain_data` has voxels."""
    return np.random.default_rng(4).standard_normal((N_IMAGES, N_REDUCED_FEATURES))


@pytest.fixture
def roi_model():
    """Common model with one feature per voxel of a single parcel."""
    return np.random.default_rng(2).standard_normal((N_IMAGES, N_VOXELS // N_PARCELS))


def mask_support(brain):
    """Count the voxels a result's own mask keeps."""
    return int(np.count_nonzero(brain.mask.get_fdata() > 0))


def assert_independent(result, *sources):
    """Every source array, mask and metadata frame must be a distinct object."""
    for source in sources:
        assert result.data is not source.data
        assert not np.shares_memory(result.data, source.data)
        assert result.mask is not source.mask
        assert result.X is not source.X
        assert result.Y is not source.Y


class TestWholeBrainContainer:
    def test_procrustes_keys_and_types(self, align_brain_data):
        out = align_brain_data.align(align_brain_data, method="procrustes")

        assert set(out) == PROCRUSTES_KEYS
        for key in SRM_KEYS:
            assert isinstance(out[key], BrainData), key
        assert isinstance(out["disparity"], float)
        assert isinstance(out["scale"], float)

    @pytest.mark.parametrize("method", SRM_METHODS)
    def test_srm_keys_and_types(self, align_brain_data, whole_brain_model, method):
        out = align_brain_data.align(whole_brain_model, method=method)

        assert set(out) == SRM_KEYS
        assert isinstance(out["transformed"], np.ndarray)
        assert isinstance(out["common_model"], np.ndarray)
        assert isinstance(out["transformation_matrix"], BrainData)

    @pytest.mark.parametrize("method", SRM_METHODS)
    def test_srm_reduced_features(self, align_brain_data, reduced_model, method):
        out = align_brain_data.align(reduced_model, method=method)

        assert isinstance(out["transformed"], np.ndarray)
        assert out["transformed"].shape == (N_IMAGES, N_REDUCED_FEATURES)
        assert isinstance(out["common_model"], np.ndarray)
        assert out["common_model"].shape == (N_IMAGES, N_REDUCED_FEATURES)

        transformation = out["transformation_matrix"]
        assert isinstance(transformation, BrainData)
        assert transformation.shape == (N_REDUCED_FEATURES, N_VOXELS)
        assert transformation.shape[-1] == mask_support(transformation)

    @pytest.mark.parametrize("method", SRM_METHODS)
    def test_srm_values_match_an_independent_solution(
        self, align_brain_data, whole_brain_model, method
    ):
        left, _, right = np.linalg.svd(
            align_brain_data.data.T.dot(whole_brain_model), full_matrices=False
        )
        expected_transformation = left.dot(right).T

        out = align_brain_data.align(whole_brain_model, method=method)

        np.testing.assert_allclose(
            out["transformation_matrix"].data, expected_transformation
        )
        np.testing.assert_allclose(
            out["transformed"], align_brain_data.data.dot(expected_transformation.T)
        )
        np.testing.assert_array_equal(out["common_model"], whole_brain_model)
        assert not np.shares_memory(out["common_model"], whole_brain_model)

    def test_procrustes_transformed_matches_the_algorithm(self, align_brain_data):
        from nltools.algorithms.alignment import procrustes

        target = align_brain_data + 1.0
        _, expected, _, rotation, _ = procrustes(target.data, align_brain_data.data)

        out = align_brain_data.align(target, method="procrustes")

        np.testing.assert_allclose(out["transformed"].data, expected)
        # The matrix is stored so that `transformed = original @ T`, which is
        # the transpose of the rotation `procrustes` solves for.
        np.testing.assert_allclose(
            out["transformation_matrix"].data, rotation.T, atol=1e-12
        )

    def test_procrustes_values_are_independent(self, align_brain_data):
        target = align_brain_data + 1.0

        out = align_brain_data.align(target, method="procrustes")

        for key in SRM_KEYS:
            assert_independent(out[key], align_brain_data, target)

    @pytest.mark.parametrize("method", SRM_METHODS)
    def test_srm_transformation_matrix_is_independent(
        self, align_brain_data, whole_brain_model, method
    ):
        out = align_brain_data.align(whole_brain_model, method=method)

        assert_independent(out["transformation_matrix"], align_brain_data)

    @pytest.mark.parametrize("method", ["procrustes"])
    def test_row_metadata_policy(self, align_brain_data, whole_brain_model, method):
        target = align_brain_data if method == "procrustes" else whole_brain_model

        out = align_brain_data.align(target, method=method)

        if method == "procrustes":
            assert out["transformed"].X.equals(align_brain_data.X)
            assert out["transformed"].Y.equals(align_brain_data.Y)
            assert out["common_model"].X.is_empty()
            assert out["common_model"].Y.is_empty()
        assert out["transformation_matrix"].X.is_empty()
        assert out["transformation_matrix"].Y.is_empty()

    @pytest.mark.parametrize("method", ["procrustes"])
    def test_every_brain_data_matches_its_mask_support(
        self, align_brain_data, whole_brain_model, method
    ):
        target = align_brain_data if method == "procrustes" else whole_brain_model

        out = align_brain_data.align(target, method=method)

        wrapped = [value for value in out.values() if isinstance(value, BrainData)]
        assert wrapped
        for value in wrapped:
            assert value.shape[-1] == mask_support(value)

    def test_target_with_more_voxels_raises(self, align_brain_data, wider_target):
        with pytest.raises(ValueError, match="mask supports 6 voxels"):
            align_brain_data.align(wider_target, method="procrustes")


class TestWholeBrainAxisOne:
    def test_procrustes(self, align_brain_data):
        out = align_brain_data.align(align_brain_data, method="procrustes", axis=1)

        assert set(out) == PROCRUSTES_KEYS
        assert isinstance(out["transformed"], BrainData)
        assert out["transformed"].shape == align_brain_data.shape
        assert isinstance(out["common_model"], BrainData)
        assert isinstance(out["transformation_matrix"], np.ndarray)
        assert out["transformation_matrix"].shape == (N_IMAGES, N_IMAGES)


class TestRoiContainer:
    def test_procrustes_keys_and_types(self, align_brain_data, align_atlas):
        out = align_brain_data.align(
            align_brain_data,
            method="procrustes",
            spatial_scale="roi",
            roi_mask=align_atlas,
        )

        assert set(out) == PROCRUSTES_KEYS | {"roi_labels"}
        assert isinstance(out["transformed"], BrainData)
        assert out["transformed"].shape == align_brain_data.shape
        for key in ("transformation_matrix", "common_model"):
            assert isinstance(out[key], dict), key
            assert list(out[key]) == [1, 2]
            for parcel in out[key].values():
                assert isinstance(parcel, BrainData)
        assert list(out["roi_labels"]) == [1, 2]
        assert out["disparity"].shape == (N_PARCELS,)
        assert out["scale"].shape == (N_PARCELS,)

    def test_srm_parcel_size_mismatch_raises(
        self, align_brain_data, align_atlas, reduced_model
    ):
        with pytest.raises(ValueError, match="cannot be painted back"):
            align_brain_data.align(
                reduced_model,
                method="deterministic_srm",
                spatial_scale="roi",
                roi_mask=align_atlas,
            )

    def test_parcel_transforms_carry_the_parcel_mask(
        self, align_brain_data, align_atlas
    ):
        from nilearn.masking import apply_mask

        label_vector = apply_mask(align_atlas, align_brain_data.mask).astype(int)

        out = align_brain_data.align(
            align_brain_data,
            method="procrustes",
            spatial_scale="roi",
            roi_mask=align_atlas,
        )

        for label, parcel in out["common_model"].items():
            n_parcel_voxels = int((label_vector == label).sum())
            assert parcel.shape[-1] == n_parcel_voxels
            assert mask_support(parcel) == n_parcel_voxels
