"""Engine-level ``device=`` kwarg — the v0.6.0 rename from ``parallel=``/``backend=``.

The canonical vocabulary (CLAUDE.md) uses ``device`` for GPU/CPU selection.
Until v0.6.0 the inference engine used ``parallel=`` (and ``backend=`` in
``phase_randomize``) with the ``nltools.stats`` facade translating names at the
boundary. With ``nltools.stats`` consolidated away, the engine itself must
speak the canonical name: ``device='cpu' | 'gpu' | None`` and a ``'device'``
key in result dicts.
"""

import inspect
import re
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


def _device_call(func):
    """Build minimal valid arguments for each device-taking entry point."""
    rng = np.random.default_rng(0)
    if func is one_sample_permutation_test:
        return (rng.standard_normal(12),), {}
    if func is isc_permutation_test:
        return (rng.standard_normal((20, 5)),), {}
    if func is isc_group_permutation_test:
        return (rng.standard_normal((20, 4)), rng.standard_normal((20, 4))), {}
    if func is phase_randomize:
        return (rng.standard_normal(16),), {}
    # two-array tests: two_sample / correlation / timeseries
    return (rng.standard_normal(20), rng.standard_normal(20)), {}


@pytest.mark.parametrize(
    "func",
    [f for f in DEVICE_FUNCTIONS if f is not matrix_permutation_test],
    ids=lambda f: f.__name__,
)
def test_invalid_device_raises_shared_message(func):
    """Every entry point rejects an invalid device with the one shared message.

    Run-or-raise (CLAUDE.md): a typo like 'gup' must never warn and silently
    run on CPU, and every entry point must use `validate_device_parameter`
    rather than a hand-rolled (drifting) copy of the check.
    """
    args, kwargs = _device_call(func)
    if func is phase_randomize:
        # phase_randomize additionally accepts 'auto' (it resolves the device
        # itself), so its message names the extra option.
        expected = re.escape("device must be None, 'cpu', 'gpu', or 'auto', got 'gup'")
    else:
        kwargs.setdefault("n_permute", 20)
        expected = re.escape("device must be None, 'cpu', or 'gpu', got 'gup'")
    with pytest.raises(ValueError, match=expected):
        func(*args, device="gup", random_state=0, **kwargs)


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

    def test_unknown_device_raises(self):
        """Run-or-raise: no warn-and-fall-back-to-CPU for a typo'd device."""
        x = np.random.default_rng(0).standard_normal(64)
        with pytest.raises(ValueError, match="device"):
            phase_randomize(x, device="quantum", random_state=0)

    def test_auto_device_accepted(self):
        x = np.random.default_rng(0).standard_normal(64)
        out = phase_randomize(x, device="auto", random_state=0)
        assert out.shape == x.shape
