"""`Glm` must fit and compute contrasts without emitting warnings.

The estimator no longer builds a `FirstLevelModel` or a masker, so nilearn's
mask-generation `RuntimeWarning` and its deprecated-accessor `FutureWarning`
have no path into nltools. `run_glm` itself is quiet at `verbose=0`.
"""

import warnings

import pytest

from nltools.models import Glm


@pytest.mark.parametrize("noise_model", ["ols", "ar1", "ar2"])
def test_fit_emits_no_warnings(glm_design, ar_targets, noise_model):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Glm(noise_model=noise_model, bins=3, random_state=0)
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
