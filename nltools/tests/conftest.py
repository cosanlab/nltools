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

import os

# Cap nested joblib/loky parallelism during tests. Collection ops default to
# n_jobs=-1; under pytest-xdist that multiplies (xdist workers x loky procs) and
# can exhaust RAM on smaller machines. setdefault keeps it overridable from the
# environment (e.g. LOKY_MAX_CPU_COUNT=8 on a big box).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import pytest
import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances
from nltools.data.simulator import Simulator
from nltools.data import Adjacency, BrainData


@pytest.fixture(scope="session", params=["2mm"])
def _sim_brain_data_source(request):
    """Expensive creation of simulated brain data — done once per session."""
    np.random.seed(0)
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 3
    dat = sim.create_data(y, sigma, reps=n_reps)
    dat.X = pl.DataFrame(
        {"Intercept": np.ones(len(dat.Y)), "X1": np.array(dat.Y).flatten()}
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
    dat.X = pl.DataFrame(
        {"Intercept": np.ones(n_samples), "X1": np.random.randn(n_samples)}
    )
    return dat


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
