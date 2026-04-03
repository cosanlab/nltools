"""BrainData-specific test fixtures."""

import pytest
import numpy as np

from nltools.data import BrainData


@pytest.fixture(scope="function")
def small_brain_data_for_cv():
    """Fast fixture for CV testing: 24 samples, 10 features, 5 voxels."""
    import nibabel as nib

    np.random.seed(42)

    spatial_shape = (3, 2, 1)
    n_samples = 24
    n_voxels = 5

    mask_data = np.zeros(spatial_shape, dtype=bool)
    mask_data.flat[:n_voxels] = True

    y_data_1d = np.random.randn(n_samples, n_voxels)

    volume_4d = np.zeros(spatial_shape + (n_samples,))
    for t in range(n_samples):
        volume_t = np.zeros(spatial_shape)
        volume_t.flat[:n_voxels] = y_data_1d[t]
        volume_4d[..., t] = volume_t

    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume_4d, affine)
    mask_img = nib.Nifti1Image(mask_data.astype(np.float32), affine)

    brain_data = BrainData(nifti_img, mask=mask_img)
    X = np.random.randn(n_samples, 10)

    return brain_data, X


@pytest.fixture(scope="function")
def tiny_brain_data_for_cv():
    """Minimal fixture for error/edge case testing: 6 samples, 5 features, 3 voxels."""
    import nibabel as nib

    np.random.seed(42)

    spatial_shape = (2, 2, 1)
    n_samples = 6
    n_voxels = 3

    mask_data = np.zeros(spatial_shape, dtype=bool)
    mask_data.flat[:n_voxels] = True

    y_data_1d = np.random.randn(n_samples, n_voxels)

    volume_4d = np.zeros(spatial_shape + (n_samples,))
    for t in range(n_samples):
        volume_t = np.zeros(spatial_shape)
        volume_t.flat[:n_voxels] = y_data_1d[t]
        volume_4d[..., t] = volume_t

    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume_4d, affine)
    mask_img = nib.Nifti1Image(mask_data.astype(np.float32), affine)

    brain_data = BrainData(nifti_img, mask=mask_img)
    X = np.random.randn(n_samples, 5)

    return brain_data, X
