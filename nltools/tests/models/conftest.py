"""Shared fixtures for model tests."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def glm_design():
    """Small full-rank single-run design carrying an explicit intercept column."""
    from nltools.data import DesignMatrix

    rng = np.random.RandomState(0)
    n_samples = 40
    return DesignMatrix(
        {
            "condition_a": rng.randn(n_samples),
            "condition_b": rng.randn(n_samples),
            "intercept": np.ones(n_samples),
        },
        sampling_freq=0.5,
    )


@pytest.fixture(scope="module")
def glm_targets(glm_design):
    """Two-dimensional response with three targets for `glm_design`."""
    rng = np.random.RandomState(1)
    return rng.randn(glm_design.shape[0], 3)


@pytest.fixture(scope="module")
def ar_targets(glm_design):
    """Temporally autocorrelated response that spreads AR coefficients over bins."""
    rng = np.random.RandomState(7)
    n_samples = glm_design.shape[0]
    noise = rng.randn(n_samples, 12)
    for row in range(1, n_samples):
        noise[row] += 0.6 * noise[row - 1]
    return noise


@pytest.fixture(scope="module")
def fitted_glm(glm_design, glm_targets):
    """OLS `_Glm` fitted on `glm_design` and the three-target response."""
    from nltools.models import _Glm

    return _Glm().fit(glm_design, glm_targets)
