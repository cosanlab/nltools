"""
Shared pytest fixtures for nltools test suite.

This module provides common fixtures used across test modules.

# Brain Data Fixtures Guide
# =========================
#
# Choose the right fixture for your test:
#
# | Fixture               | Voxels | Samples | Use Case                          |
# |-----------------------|--------|---------|-----------------------------------|
# | sim_brain_data        | Full   | 6       | Realistic brain, slow but thorough|
# | minimal_brain_data    | 5      | 50      | API contract tests, fast          |
# | small_brain_data_for_cv| 5     | 24      | CV tests (24 divisible by 3)      |
# | tiny_brain_data_for_cv | 3     | 6       | Edge cases, insufficient samples  |
#
# Use minimal_brain_data for most tests - it's 10x faster than sim_brain_data.
# Use sim_brain_data only when you need realistic brain structure.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from nltools.simulator import Simulator
from nltools.data import Adjacency, DesignMatrix, BrainData
import os
import importlib.util


# ============================================================================
# Optional Dependency Checks
# ============================================================================


def _pybids_available():
    """Check if pybids is installed."""
    return importlib.util.find_spec("bids") is not None


HAS_PYBIDS = _pybids_available()


def _tables_available():
    """Check if PyTables is installed (for HDF5 support via pandas).

    Note: HDF5 support is being deprecated in favor of more modern formats.
    Tests requiring PyTables will be skipped when it's not available.
    """
    return importlib.util.find_spec("tables") is not None


@pytest.fixture(scope="session", params=["2mm"])
def _sim_brain_data_source(request):
    """Expensive creation of simulated brain data — done once per session."""
    np.random.seed(0)
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 3
    dat = sim.create_data(y, sigma, reps=n_reps)
    dat.X = pd.DataFrame(
        {"Intercept": np.ones(len(dat.Y)), "X1": np.array(dat.Y).flatten()}, index=None
    )
    return dat


@pytest.fixture()
def sim_brain_data(_sim_brain_data_source):
    """Fresh deep copy of simulated brain data for each test."""
    return _sim_brain_data_source.copy()


@pytest.fixture(scope="function")
def minimal_brain_data():
    """Minimal BrainData for fast API contract testing.

    Creates BrainData with:
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

    # Create BrainData
    dat = BrainData(nifti_img, mask=mask_img)
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
    return DesignMatrix(
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


# ============================================================================
# H5 Test Data Path Fixtures
# ============================================================================
# Helper to reduce path resolution boilerplate


def _get_test_data_path(request, filename):
    """Get path to test data file in nltools/tests/data/."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(tests_dir, "data", filename)


@pytest.fixture(scope="module")
def old_h5_brain(request):
    """Path to old-format brain H5 file."""
    return _get_test_data_path(request, "old_brain.h5")


@pytest.fixture(scope="module")
def new_h5_brain(request):
    """Path to new-format brain H5 file."""
    return _get_test_data_path(request, "new_brain.h5")


@pytest.fixture(scope="module")
def old_h5_adj_single(request):
    """Path to old-format single adjacency H5 file."""
    return _get_test_data_path(request, "old_single.h5")


@pytest.fixture(scope="module")
def new_h5_adj_single(request):
    """Path to new-format single adjacency H5 file."""
    return _get_test_data_path(request, "new_single.h5")


@pytest.fixture(scope="module")
def old_h5_adj_double(request):
    """Path to old-format double adjacency H5 file."""
    return _get_test_data_path(request, "old_double.h5")


@pytest.fixture(scope="module")
def new_h5_adj_double(request):
    """Path to new-format double adjacency H5 file."""
    return _get_test_data_path(request, "new_double.h5")


# ============================================================================
# BIDS Dataset Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def minimal_bids_dataset(tmp_path):
    """Create minimal BIDS structure with small nifti files for testing.

    Creates a valid BIDS dataset structure with:
    - 2 subjects (sub-01, sub-02)
    - 1 task (rest)
    - Minimal 4D nifti files with valid headers

    This fixture creates real (but tiny) nifti files that pybids can parse.
    The files are small enough to be fast but valid enough for BIDSLayout.

    Returns:
        Path to the temporary BIDS dataset root directory.
    """
    import nibabel as nib
    import json

    # Create BIDS root structure
    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir()

    # Create dataset_description.json (required for valid BIDS)
    dataset_description = {
        "Name": "Test Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
    }
    with open(bids_root / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    # Create minimal 4D data for nifti files
    # Small enough to be fast, valid enough for pybids
    spatial_shape = (3, 3, 3)
    n_timepoints = 10
    affine = np.eye(4) * 2  # 2mm voxels
    affine[3, 3] = 1

    # Create subjects
    for sub_id in ["01", "02"]:
        sub_dir = bids_root / f"sub-{sub_id}" / "func"
        sub_dir.mkdir(parents=True)

        # Create 4D bold data
        bold_data = np.random.randn(*spatial_shape, n_timepoints).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, affine)

        # Save bold file with BIDS naming
        bold_path = sub_dir / f"sub-{sub_id}_task-rest_bold.nii.gz"
        nib.save(bold_img, bold_path)

        # Create sidecar JSON (optional but good practice)
        sidecar = {"TaskName": "rest", "RepetitionTime": 2.0}
        with open(sub_dir / f"sub-{sub_id}_task-rest_bold.json", "w") as f:
            json.dump(sidecar, f)

    return bids_root


@pytest.fixture(scope="function")
def minimal_bids_mask(tmp_path):
    """Create a minimal brain mask compatible with minimal_bids_dataset.

    Returns a nibabel Nifti1Image mask matching the spatial dimensions
    of the minimal_bids_dataset fixture.
    """
    import nibabel as nib

    spatial_shape = (3, 3, 3)
    affine = np.eye(4) * 2
    affine[3, 3] = 1

    # Create mask with all voxels active
    mask_data = np.ones(spatial_shape, dtype=np.float32)
    mask_img = nib.Nifti1Image(mask_data, affine)

    return mask_img
