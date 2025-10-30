import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from nltools.simulator import Simulator
from nltools.data import Adjacency, Design_Matrix, Brain_Data
import os


@pytest.fixture(scope="module", params=["2mm"])
def sim_brain_data():
    np.random.seed(0)
    # MNI_Template["resolution"] = request.params
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 3
    dat = sim.create_data(y, sigma, reps=n_reps)
    dat.X = pd.DataFrame(
        {"Intercept": np.ones(len(dat.Y)), "X1": np.array(dat.Y).flatten()}, index=None
    )
    return dat


@pytest.fixture(scope="function")
def minimal_brain_data():
    """Minimal Brain_Data for fast API contract testing.

    Creates Brain_Data with:
    - 5 active voxels (minimal spatial structure)
    - 50 timepoints (sufficient for most operations including filtering)
    - Random data (seeded for reproducibility)

    Use this fixture for testing API contracts (parameters, return types,
    shape preservation) where computational correctness is handled by
    dependencies (nilearn, sklearn, etc.).

    Performance: ~1-2s per test vs ~16s with full brain data (238,955 voxels).

    Examples:
        - Parameter validation tests
        - Error handling tests
        - Return type checks
        - Shape preservation tests
        - Fast smoke tests

    For tests requiring realistic brain structure or specific voxel counts,
    use `sim_brain_data` or create custom fixtures.
    """
    import nibabel as nib

    np.random.seed(42)

    # Minimal 3D volume: 5 active voxels
    spatial_shape = (3, 2, 1)
    n_samples = 50
    n_voxels = 5

    # Create mask
    mask_data = np.zeros(spatial_shape, dtype=bool)
    mask_data.flat[:n_voxels] = True

    # Create random timeseries data
    y_data_1d = np.random.randn(n_samples, n_voxels)

    # Build 4D volume
    volume_4d = np.zeros(spatial_shape + (n_samples,))
    for t in range(n_samples):
        volume_t = np.zeros(spatial_shape)
        volume_t.flat[:n_voxels] = y_data_1d[t]
        volume_4d[..., t] = volume_t

    # Create nibabel images
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume_4d, affine)
    mask_img = nib.Nifti1Image(mask_data.astype(np.float32), affine)

    # Create Brain_Data
    dat = Brain_Data(nifti_img, mask=mask_img)
    dat.X = pd.DataFrame(
        {"Intercept": np.ones(n_samples), "X1": np.random.randn(n_samples)}, index=None
    )
    return dat


@pytest.fixture(scope="module")
def sim_design_matrix():
    np.random.seed(0)
    # Design matrices are specified in terms of sampling frequency
    TR = 2.0
    sampling_freq = 1.0 / TR
    return Design_Matrix(
        np.random.randint(2, size=(500, 4)),
        columns=["face_A", "face_B", "house_A", "house_B"],
        sampling_freq=sampling_freq,
    )


@pytest.fixture(scope="module")
def sim_adjacency_single():
    np.random.seed(0)
    # Create a positive definite covariance matrix
    cov_matrix = np.array(
        [
            [1.0, 0.5, 0.1, 0.2],
            [0.5, 1.0, 0.3, 0.1],
            [0.1, 0.3, 1.0, 0.2],
            [0.2, 0.1, 0.2, 1.0],
        ]
    )
    sim = np.random.multivariate_normal([0, 0, 0, 0], cov_matrix, 100)
    data = pairwise_distances(sim.T, metric="correlation")
    labels = ["v_%s" % (x + 1) for x in range(sim.shape[1])]
    return Adjacency(data, labels=labels)


@pytest.fixture(scope="module")
def sim_adjacency_multiple():
    np.random.seed(0)
    n = 10
    # Create a positive definite covariance matrix
    cov_matrix = np.array(
        [
            [1.0, 0.5, 0.1, 0.2],
            [0.5, 1.0, 0.3, 0.1],
            [0.1, 0.3, 1.0, 0.2],
            [0.2, 0.1, 0.2, 1.0],
        ]
    )
    sim = np.random.multivariate_normal([0, 0, 0, 0], cov_matrix, 100)
    data = pairwise_distances(sim.T, metric="correlation")
    dat_all = []
    for t in range(n):
        tmp = data
        dat_all.append(tmp)
    labels = ["v_%s" % (x + 1) for x in range(sim.shape[1])]
    return Adjacency(dat_all, labels=labels)


@pytest.fixture(scope="module")
def sim_adjacency_directed():
    sim_directed = np.array(
        [
            [1, 0.5, 0.3, 0.4],
            [0.8, 1, 0.2, 0.1],
            [0.7, 0.6, 1, 0.5],
            [0.85, 0.4, 0.3, 1],
        ]
    )
    labels = ["v_%s" % (x + 1) for x in range(sim_directed.shape[1])]
    return Adjacency(sim_directed, matrix_type="directed", labels=labels)


@pytest.fixture(scope="module")
def old_h5_brain(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "old_brain.h5")


@pytest.fixture(scope="module")
def new_h5_brain(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "new_brain.h5")


@pytest.fixture(scope="module")
def old_h5_adj_single(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "old_single.h5")


@pytest.fixture(scope="module")
def new_h5_adj_single(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "new_single.h5")


@pytest.fixture(scope="module")
def old_h5_adj_double(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "old_double.h5")


@pytest.fixture(scope="module")
def new_h5_adj_double(request):
    # Navigate from test file location to data directory
    test_file_dir = os.path.dirname(request.module.__file__)
    tests_dir = os.path.dirname(test_file_dir)
    return os.path.join(tests_dir, "data", "new_double.h5")


@pytest.fixture(scope="module")
def regress_result(sim_brain_data):
    # Create labels based on actual data shape
    n_conditions = sim_brain_data.shape[0]
    labels = ["condition_" + str(i) for i in range(n_conditions)]
    # Make face and house special indices for testing
    if n_conditions >= 4:
        labels[3] = "face"
    if n_conditions >= 5:
        labels[4] = "house"

    # 64 "TRs"
    fake_timeseries = Brain_Data([sim_brain_data] * 8)

    return {
        "z_score": sim_brain_data,
        "t": sim_brain_data.copy(),
        "p": sim_brain_data.copy(),
        "beta": sim_brain_data.copy(),
        "se": sim_brain_data.copy(),
        "rsquared": sim_brain_data.copy()[0],  # 1 value per voxel
        "residual": fake_timeseries - fake_timeseries.mean(),
        "predicted": fake_timeseries,
        "labels": labels,
    }


@pytest.fixture(scope="function")
def small_brain_data_for_cv():
    """Fast fixture for CV testing: 24 samples, 10 features, 5 voxels.

    Designed for integration tests - small enough to run in <0.1s per test.
    24 samples divisible by 3 for clean 3-fold CV.
    """
    import nibabel as nib

    np.random.seed(42)

    # Create synthetic brain data with 5 voxels arranged in a tiny 3D volume
    # Shape: (2, 2, 2) spatial dimensions with 1 non-zero voxel gives us flexibility
    # We'll create 24 timepoints (samples) with 5 active voxels
    # Use a simple mask approach
    spatial_shape = (3, 2, 1)  # Small 3D volume with 6 voxels total
    n_samples = 24
    n_voxels = 5

    # Create 4D data: (3, 2, 1, 24) but we'll only use 5 voxels
    # Create mask: first 5 voxels are active
    mask_data = np.zeros(spatial_shape, dtype=bool)
    mask_data.flat[:n_voxels] = True

    # Create random data for active voxels
    y_data_1d = np.random.randn(n_samples, n_voxels)

    # Construct 4D volume
    volume_4d = np.zeros(spatial_shape + (n_samples,))
    for t in range(n_samples):
        volume_t = np.zeros(spatial_shape)
        volume_t.flat[:n_voxels] = y_data_1d[t]
        volume_4d[..., t] = volume_t

    # Create nibabel image with identity affine
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume_4d, affine)
    mask_img = nib.Nifti1Image(mask_data.astype(np.float32), affine)

    # Create Brain_Data from nibabel image with mask
    brain_data = Brain_Data(nifti_img, mask=mask_img)

    # Create corresponding features
    X = np.random.randn(n_samples, 10)

    return brain_data, X


@pytest.fixture(scope="function")
def tiny_brain_data_for_cv():
    """Minimal fixture for error/edge case testing: 6 samples, 5 features, 3 voxels.

    Used for testing insufficient samples errors and validation logic.
    """
    import nibabel as nib

    np.random.seed(42)

    # Small 3D volume
    spatial_shape = (2, 2, 1)
    n_samples = 6
    n_voxels = 3

    # Create mask: first 3 voxels active
    mask_data = np.zeros(spatial_shape, dtype=bool)
    mask_data.flat[:n_voxels] = True

    # Create random data
    y_data_1d = np.random.randn(n_samples, n_voxels)

    # Construct 4D volume
    volume_4d = np.zeros(spatial_shape + (n_samples,))
    for t in range(n_samples):
        volume_t = np.zeros(spatial_shape)
        volume_t.flat[:n_voxels] = y_data_1d[t]
        volume_4d[..., t] = volume_t

    # Create nibabel images
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume_4d, affine)
    mask_img = nib.Nifti1Image(mask_data.astype(np.float32), affine)

    # Create Brain_Data
    brain_data = Brain_Data(nifti_img, mask=mask_img)
    X = np.random.randn(n_samples, 5)

    return brain_data, X
