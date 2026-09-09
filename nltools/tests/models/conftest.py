"""Shared fixtures for model tests."""

import numpy as np
import pytest


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
