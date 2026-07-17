"""Fast API-convention tests for the alignment estimators.

These pin keyword-only signatures (F005) and the progress_bar default (F004)
without running the (slow) alignment algorithms.
"""

import numpy as np
import pytest

from nltools.algorithms.alignment import SRM, DetSRM, HyperAlignment, LocalAlignment


# ========== F005: keyword-only `*` marker on public estimator methods ==========


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_fit_rejects_positional_options(cls):
    """fit() must reject parallel passed positionally (binding fails pre-compute)."""
    X = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        cls().fit(X, None, "cpu")


@pytest.mark.parametrize("cls", [SRM, DetSRM])
def test_srm_transform_rejects_positional_options(cls):
    """transform() must reject parallel passed positionally."""
    X = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        cls().transform(X, None, "cpu")


def test_hyperalignment_fit_rejects_positional_options():
    """HyperAlignment.fit() must reject parallel passed positionally."""
    data = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        HyperAlignment().fit(data, "cpu")


def test_hyperalignment_transform_rejects_positional_options():
    """HyperAlignment.transform() must reject parallel passed positionally."""
    data = [np.zeros((3, 3)), np.zeros((3, 3))]
    with pytest.raises(TypeError):
        HyperAlignment().transform(data, "cpu")


# ========== F004: LocalAlignment progress_bar defaults to off ==========


def test_localalignment_progress_bar_defaults_off():
    """LocalAlignment must expose progress_bar defaulting to False."""
    la = LocalAlignment()
    assert la.progress_bar is False
