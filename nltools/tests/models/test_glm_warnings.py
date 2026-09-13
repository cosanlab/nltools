"""`_Glm` must fit and compute contrasts without emitting warnings.

The estimator no longer builds a `FirstLevelModel` or a masker, so nilearn's
mask-generation `RuntimeWarning` and its deprecated-accessor `FutureWarning`
have no path into nltools. `run_glm` itself is quiet at `verbose=0`.
"""

import warnings

import pytest

from nltools.models import _Glm


@pytest.mark.parametrize("noise_model", ["ols", "ar1", "ar2"])
def test_fit_emits_no_warnings(glm_design, ar_targets, noise_model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = _Glm(noise_model=noise_model, bins=3, random_state=0)
        model.fit(glm_design, ar_targets)
    assert model.is_fitted_


def test_contrasts_emit_no_warnings(fitted_glm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted_glm.compute_contrasts("condition_a - condition_b")
        fitted_glm.compute_contrasts([1, -1, 0], inference=True)


def test_predict_emits_no_warnings(glm_design, fitted_glm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted_glm.predict(glm_design)


def test_a_constant_target_fits_without_warning(glm_design, glm_targets):
    """A voxel with no variance has an undefined R-squared, not a warning.

    Nilearn divides the fitted variance by the target's own variance, so a
    constant target — an empty voxel inside a mask, which any real brain mask
    contains — makes that ratio 0/0 or x/0. Copying the value must not emit
    numpy's `RuntimeWarning`.
    """
    import numpy as np

    targets = np.array(glm_targets, copy=True)
    targets[:, 0] = 3.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = _Glm().fit(glm_design, targets)

    assert not np.isfinite(model.r2_[0])
    assert np.isfinite(model.r2_[1:]).all()
