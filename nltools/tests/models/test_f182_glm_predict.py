"""F182: `Glm.predict` applies the fitted coefficients to a new design.

The original finding was a `predict(X)` that documented a new-design path and
then raised. Under docs/development/specs/glm.md the method takes a
`DesignMatrix`, matches it to the fitted column names, and returns
`X @ coef_` in observation space.
"""

import numpy as np
import pytest

from nltools.data import DesignMatrix
from nltools.models import Glm


@pytest.fixture
def new_design(glm_design):
    """A different set of observations over the fitted regressor names."""
    rng = np.random.RandomState(11)
    n_samples = 8
    return DesignMatrix(
        {
            "condition_a": rng.randn(n_samples),
            "condition_b": rng.randn(n_samples),
            "intercept": np.ones(n_samples),
        },
        sampling_freq=glm_design.sampling_freq,
    )


def test_new_design_prediction_equals_design_at_coef(fitted_glm, new_design):
    predictions = fitted_glm.predict(new_design)
    assert predictions.shape == (8, fitted_glm.n_targets_)
    np.testing.assert_allclose(predictions, new_design.to_numpy() @ fitted_glm.coef_)


def test_column_order_does_not_change_the_prediction(fitted_glm, new_design):
    shuffled = new_design[["intercept", "condition_a", "condition_b"]]
    np.testing.assert_allclose(
        fitted_glm.predict(shuffled), fitted_glm.predict(new_design)
    )


def test_predict_takes_only_a_design_matrix(fitted_glm, new_design):
    with pytest.raises(TypeError, match="DesignMatrix"):
        fitted_glm.predict(new_design.to_numpy())


def test_predict_before_fit_raises(new_design):
    with pytest.raises(ValueError, match="not fitted"):
        Glm().predict(new_design)
