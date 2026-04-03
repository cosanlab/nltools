"""Tests for Glm model: instantiation, fit, contrasts, properties, and scoring."""

import warnings

import numpy as np
import pandas as pd
import pytest
from nibabel import Nifti1Image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

from nltools.models import Glm

pytestmark = pytest.mark.slow


class TestGlmCore:
    """Instantiation, fitting, and basic state tracking."""

    def test_instantiation(self):
        """Glm should instantiate with optional parameters."""
        model = Glm()
        assert not model.is_fitted_

        model = Glm(t_r=2.0, noise_model="ar1", smoothing_fwhm=5.0)
        assert model.t_r == 2.0
        assert model.noise_model == "ar1"
        assert model.smoothing_fwhm == 5.0

    def test_requires_nilearn(self):
        """Glm should expose fit method (nilearn dependency present)."""
        model = Glm()
        assert hasattr(model, "fit")

    def test_fit_returns_self(self, glm_single_run_data):
        """fit() should return self and mark model as fitted."""
        model = Glm(t_r=2.0, mask=glm_single_run_data["mask_img"])
        result = model.fit(
            glm_single_run_data["img"],
            design_matrices=glm_single_run_data["design_matrix"],
        )
        assert result is model
        assert model.is_fitted_

    def test_fit_tracks_state(self):
        """Glm should track fitted state."""
        model = Glm()
        assert not model.is_fitted_

    def test_suppresses_drift_model_warning(self, glm_single_run_data):
        """Glm should suppress drift_model warning when design matrices are supplied."""
        model = Glm(t_r=2.0, mask=glm_single_run_data["mask_img"], drift_model="cosine")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(
                glm_single_run_data["img"],
                design_matrices=glm_single_run_data["design_matrix"],
            )

            drift_warnings = [
                warn
                for warn in w
                if "drift_model" in str(warn.message).lower()
                and "will be ignored" in str(warn.message).lower()
            ]
            assert len(drift_warnings) == 0

        assert model.is_fitted_
        assert model.progress_bar is False

        # progress_bar=True should also work
        model_with_pb = Glm(
            t_r=2.0, mask=glm_single_run_data["mask_img"], progress_bar=True
        )
        assert model_with_pb.progress_bar is True
        model_with_pb.fit(
            glm_single_run_data["img"],
            design_matrices=glm_single_run_data["design_matrix"],
        )
        assert model_with_pb.is_fitted_

    def test_multiple_runs(self):
        """Glm should handle multiple runs."""
        np.random.seed(42)
        n_scans = 20
        img_shape = (10, 10, 10)
        affine = np.eye(4)
        mask_img = Nifti1Image(np.ones(img_shape, dtype=np.int8), affine)

        images = []
        design_matrices = []
        for _ in range(2):
            fmri_data = np.random.randn(n_scans, *img_shape).astype(np.float32)
            images.append(Nifti1Image(fmri_data.T, affine))

            frame_times = np.arange(n_scans) * 2.0
            events = pd.DataFrame(
                {"onset": [0, 10], "duration": [1, 1], "trial_type": ["task", "task"]}
            )
            design_matrices.append(
                make_first_level_design_matrix(
                    frame_times, events=events, hrf_model="spm"
                )
            )

        model = Glm(t_r=2.0, mask=mask_img)
        model.fit(images, design_matrices=design_matrices)

        assert model.is_fitted_
        assert model.progress_bar is False

        contrast_map = model.compute_contrast("task")
        assert hasattr(contrast_map, "get_fdata")


class TestGlmContrasts:
    """Contrast computation after fitting."""

    def test_compute_contrast(self, fitted_glm_single_run):
        """compute_contrast should return a Nifti image with correct shape."""
        model, data = fitted_glm_single_run
        contrast_map = model.compute_contrast("task")

        assert hasattr(contrast_map, "get_fdata")
        assert contrast_map.shape == data["img_shape"]

    def test_compute_contrast_before_fit_raises(self):
        """compute_contrast should raise error before fit."""
        model = Glm()
        with pytest.raises(ValueError, match="not fitted"):
            model.compute_contrast("task")


class TestGlmProperties:
    """Property access: residuals, design_matrices_, glm_, score."""

    def test_residuals(self, fitted_glm_single_run):
        """Glm should expose residuals as list of Nifti images."""
        model, _ = fitted_glm_single_run

        residuals = model.residuals
        assert isinstance(residuals, list)
        assert len(residuals) == 1
        assert hasattr(residuals[0], "get_fdata")
        assert model.progress_bar is False

    def test_design_matrices(self, fitted_glm_single_run):
        """Glm should expose design_matrices_ with correct columns."""
        model, _ = fitted_glm_single_run

        design_mats = model.design_matrices_
        assert isinstance(design_mats, list)
        assert len(design_mats) == 1
        assert "task" in design_mats[0].columns

    def test_glm_property(self):
        """Glm.glm_ should expose the internal FirstLevelModel."""
        model = Glm(t_r=2.0)
        assert isinstance(model.glm_, FirstLevelModel)
        assert model.glm_ is model._glm

    def test_score_returns_valid_r_squared(self):
        """Glm.score() should return mean R-squared in [0, 1]."""
        np.random.seed(42)
        n_scans = 30
        img_shape = (8, 8, 8)

        # Create data with actual task signal
        task_signal = np.zeros(n_scans)
        task_signal[5:10] = 1.0
        task_signal[20:25] = 1.0

        fmri_data = np.zeros((n_scans, *img_shape), dtype=np.float32)
        for t in range(n_scans):
            signal_voxels = fmri_data[t, :4, :4, :4]
            signal_voxels += task_signal[t] * 2.0
            fmri_data[t] += np.random.randn(*img_shape).astype(np.float32) * 0.5

        affine = np.eye(4)
        img = Nifti1Image(fmri_data.T, affine)
        mask_img = Nifti1Image(np.ones(img_shape, dtype=np.int8), affine)

        frame_times = np.arange(n_scans) * 2.0
        events = pd.DataFrame(
            {
                "onset": [10, 40],
                "duration": [10, 10],
                "trial_type": ["task", "task"],
            }
        )
        design_matrix = make_first_level_design_matrix(
            frame_times, events=events, hrf_model="spm"
        )

        model = Glm(t_r=2.0, mask=mask_img)
        model.fit(img, design_matrices=design_matrix)

        r2 = model.score()
        assert isinstance(r2, float)
        assert 0.0 < r2 < 1.0
        assert r2 > 0.01

    def test_score_before_fit_raises(self):
        """score() should raise error before fit."""
        model = Glm(t_r=2.0)
        with pytest.raises(ValueError, match="not fitted yet"):
            model.score()

    def test_score_sklearn_api_compatibility(self, fitted_glm_single_run):
        """score() should accept X and y kwargs for sklearn compatibility."""
        model, _ = fitted_glm_single_run

        r2_no_args = model.score()
        r2_with_args = model.score(X=None, y=None)
        assert r2_no_args == r2_with_args
