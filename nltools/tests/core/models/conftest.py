"""Shared fixtures for model tests."""

import importlib.util

import numpy as np
import pytest

from nltools.models import Ridge


def torch_available():
    """Check if PyTorch is installed."""
    return importlib.util.find_spec("torch") is not None


@pytest.fixture(scope="module")
def ridge_single_target_data():
    """Standard single-target Ridge test data.

    Module-scoped: deterministic data shared across tests.
    """
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)
    return {"X": X, "y": y, "X_test": X_test}


@pytest.fixture(scope="module")
def ridge_multi_target_data():
    """Standard multi-target Ridge test data."""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)
    X_test = np.random.randn(20, 50).astype(np.float32)
    return {"X": X, "Y": Y, "X_test": X_test}


@pytest.fixture(scope="module")
def fitted_ridge_single(ridge_single_target_data):
    """Pre-fitted Ridge model for property tests.

    Module-scoped: expensive fit() runs once.
    """
    model = Ridge(alpha=1.0)
    model.fit(ridge_single_target_data["X"], ridge_single_target_data["y"])
    return model, ridge_single_target_data


@pytest.fixture(scope="module")
def fitted_ridge_cv(ridge_single_target_data):
    """Pre-fitted Ridge with CV for property tests."""
    model = Ridge(alpha="auto", cv=3)
    model.fit(ridge_single_target_data["X"], ridge_single_target_data["y"])
    return model, ridge_single_target_data


@pytest.fixture(scope="module")
def glm_single_run_data():
    """Synthetic single-run fMRI data for GLM tests.

    Module-scoped: expensive setup shared across tests.
    """
    from nilearn.glm.first_level import make_first_level_design_matrix
    import pandas as pd
    from nibabel import Nifti1Image

    np.random.seed(42)
    n_scans = 20
    img_shape = (10, 10, 10)
    fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
    affine = np.eye(4)
    img = Nifti1Image(fmri_data.T, affine)

    mask_data = np.ones(img_shape, dtype=np.int8)
    mask_img = Nifti1Image(mask_data, affine)

    frame_times = np.arange(n_scans) * 2.0
    events = pd.DataFrame(
        {"onset": [0, 10], "duration": [1, 1], "trial_type": ["task", "task"]}
    )
    design_matrix = make_first_level_design_matrix(
        frame_times, events=events, hrf_model="spm"
    )

    return {
        "img": img,
        "mask_img": mask_img,
        "design_matrix": design_matrix,
        "img_shape": img_shape,
        "n_scans": n_scans,
    }


@pytest.fixture(scope="module")
def fitted_glm_single_run(glm_single_run_data):
    """Pre-fitted GLM on single-run data."""
    from nltools.models import Glm

    model = Glm(t_r=2.0, mask=glm_single_run_data["mask_img"])
    model.fit(
        glm_single_run_data["img"],
        design_matrices=glm_single_run_data["design_matrix"],
    )
    return model, glm_single_run_data
