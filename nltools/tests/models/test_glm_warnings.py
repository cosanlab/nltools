"""Glm must not trip nilearn's deprecation or mask-generation warnings.

Two causes, both in ``Glm``'s glue to ``FirstLevelModel``:

- ``residuals`` / ``predicted`` / ``r_square`` are deprecated accessors in
  nilearn >= 0.14 (``FutureWarning``); the fitted attributes carry a trailing
  underscore.
- Passing a ``Nifti1Image`` as ``mask_img`` makes nilearn build a
  ``MultiNiftiMasker`` and *fit it on the run images*, which warns
  (``RuntimeWarning``: "Generation of a mask has been requested ... while a
  mask was given") even though the given mask is what ends up used. A
  pre-fitted ``NiftiMasker`` is used as-is, silently.
"""

import warnings

import numpy as np
import pytest
from nilearn.glm.first_level import FirstLevelModel

from nltools.models import Glm


@pytest.fixture(scope="module")
def fitted_glm(glm_single_run_data):
    """Fit once, in the warning regime the tests assert about."""
    model = Glm(t_r=2.0, mask=glm_single_run_data["mask_img"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(
            glm_single_run_data["img"],
            design_matrices=glm_single_run_data["design_matrix"],
        )
    return model


class TestNoDeprecatedAccessors:
    def test_residuals_predicted_score_emit_no_future_warning(self, fitted_glm):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            residuals = fitted_glm.residuals
            predicted = fitted_glm.predict()
            r2 = fitted_glm.score()
        assert len(residuals) == 1
        assert len(predicted) == 1
        assert np.isfinite(r2)


class TestNoMaskGenerationWarning:
    def test_fit_emits_no_runtime_warning(self, glm_single_run_data):
        model = Glm(t_r=2.0, mask=glm_single_run_data["mask_img"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.fit(
                glm_single_run_data["img"],
                design_matrices=glm_single_run_data["design_matrix"],
            )
        assert model.is_fitted_

    def test_betas_match_nilearn_with_nifti_mask(self, fitted_glm, glm_single_run_data):
        """The pre-fitted masker changes nothing numerically."""
        reference = FirstLevelModel(
            t_r=2.0,
            noise_model="ols",
            mask_img=glm_single_run_data["mask_img"],
            minimize_memory=False,
            standardize=False,
            signal_scaling=False,
            drift_model=None,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference.fit(
                glm_single_run_data["img"],
                design_matrices=glm_single_run_data["design_matrix"],
            )
        ref_betas = np.zeros_like(fitted_glm.coef_)
        labels = reference.labels_[0]
        for lab, res in reference.results_[0].items():
            ref_betas[:, labels == lab] = res.theta
        np.testing.assert_allclose(fitted_glm.coef_, ref_betas)

    def test_masker_carries_glm_smoothing_and_tr(self):
        """nilearn does not copy smoothing_fwhm/t_r onto a user masker, so Glm must."""
        import nibabel as nib

        mask = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int8), np.eye(4))
        model = Glm(t_r=1.5, smoothing_fwhm=6.0, mask=mask)
        masker = model.glm_.mask_img
        assert masker.smoothing_fwhm == 6.0
        assert masker.t_r == 1.5
        assert masker.mask_img is mask
