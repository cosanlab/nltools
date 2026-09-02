"""nltools.algorithms.hrf wraps nilearn's HRFs with keyword-only options."""

import inspect

import numpy as np
import pytest
from nilearn.glm import first_level as nilearn_hrf

from nltools.algorithms import hrf

WRAPPERS = [
    "glover_hrf",
    "glover_time_derivative",
    "glover_dispersion_derivative",
    "spm_hrf",
    "spm_time_derivative",
    "spm_dispersion_derivative",
]


@pytest.mark.parametrize("name", WRAPPERS)
def test_matches_nilearn(name):
    ours = getattr(hrf, name)(2.0, oversampling=10, time_length=20.0, onset=1.0)
    theirs = getattr(nilearn_hrf, name)(2.0, 10, 20.0, 1.0)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("name", WRAPPERS)
def test_options_are_keyword_only(name):
    params = inspect.signature(getattr(hrf, name)).parameters
    assert list(params) == ["t_r", "oversampling", "time_length", "onset"]
    assert all(
        params[p].kind is inspect.Parameter.KEYWORD_ONLY
        for p in ("oversampling", "time_length", "onset")
    )
    with pytest.raises(TypeError):
        getattr(hrf, name)(2.0, 10)


def test_all_exports_documented():
    assert sorted(hrf.__all__) == sorted(WRAPPERS)
