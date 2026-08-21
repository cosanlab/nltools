"""Engine-level ``device=`` kwarg — the v0.6.0 rename from ``parallel=``/``backend=``.

The canonical vocabulary (CLAUDE.md) uses ``device`` for GPU/CPU selection.
Until v0.6.0 the inference engine used ``parallel=`` (and ``backend=`` in
``phase_randomize``) with the ``nltools.stats`` facade translating names at the
boundary. With ``nltools.stats`` consolidated away, the engine itself must
speak the canonical name: ``device='cpu' | 'gpu' | None`` and a ``'device'``
key in result dicts.
"""

import inspect
import warnings

import numpy as np
import pytest

from nltools.algorithms.inference import (
    correlation_permutation_test,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    phase_randomize,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)

DEVICE_FUNCTIONS = [
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    matrix_permutation_test,
    timeseries_correlation_permutation_test,
    isc_permutation_test,
    isc_group_permutation_test,
    phase_randomize,
]


@pytest.mark.parametrize("func", DEVICE_FUNCTIONS, ids=lambda f: f.__name__)
def test_signature_uses_device_not_parallel_or_backend(func):
    params = inspect.signature(func).parameters
    assert "device" in params, f"{func.__name__} lacks the canonical device kwarg"
    assert "parallel" not in params, f"{func.__name__} still exposes parallel="
    assert "backend" not in params, f"{func.__name__} still exposes backend="
    assert params["device"].kind is inspect.Parameter.KEYWORD_ONLY


def test_result_dict_reports_device_not_parallel():
    rng = np.random.default_rng(0)
    result = one_sample_permutation_test(
        rng.standard_normal(12), n_permute=20, device="cpu", n_jobs=1, random_state=0
    )
    assert result["device"] == "cpu"
    assert "parallel" not in result


def test_single_threaded_device_none_reports_none():
    rng = np.random.default_rng(0)
    result = one_sample_permutation_test(
        rng.standard_normal(12), n_permute=20, device=None, random_state=0
    )
    assert result["device"] is None


def test_invalid_device_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="device"):
        one_sample_permutation_test(
            rng.standard_normal(12), n_permute=20, device="tpu", random_state=0
        )


def test_matrix_rejects_gpu_device():
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((6, 6))
    mat = mat + mat.T
    with pytest.raises(ValueError, match="device"):
        matrix_permutation_test(mat, mat, n_permute=20, device="gpu", random_state=0)


class TestPhaseRandomizeDevice:
    def test_cpu_device(self):
        x = np.random.default_rng(0).standard_normal(64)
        out = phase_randomize(x, device="cpu", random_state=0)
        assert out.shape == x.shape
        # Power spectrum preserved
        np.testing.assert_allclose(
            np.abs(np.fft.fft(out)), np.abs(np.fft.fft(x)), rtol=1e-8
        )

    def test_default_is_cpu_and_silent(self):
        x = np.random.default_rng(0).standard_normal(64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = phase_randomize(x, random_state=0)
        assert out.shape == x.shape

    def test_unknown_device_warns_and_falls_back(self):
        x = np.random.default_rng(0).standard_normal(64)
        with pytest.warns(UserWarning, match="device"):
            out = phase_randomize(x, device="quantum", random_state=0)
        assert out.shape == x.shape
