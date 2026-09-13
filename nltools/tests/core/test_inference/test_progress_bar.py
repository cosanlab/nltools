"""
Tests for the ``progress_bar`` keyword across the inference family.

Progress bars are written to stderr by tqdm. A library should not write to
stderr unless asked: calling these functions in a loop (e.g. a calibration
simulation running 100 permutation tests) otherwise emits one progress bar per
call. Per the canonical kwarg table, the knob is ``progress_bar: bool = False``.
"""

import contextlib
import inspect
import io

import numpy as np
import pytest

from nltools.algorithms.inference import (
    correlation_permutation_test,
    _isc_group_permutation_test,
    _isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    _timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)

# Every algorithm-layer entry point that drives a tqdm loop.
PROGRESS_BAR_FUNCTIONS = [
    correlation_permutation_test,
    _isc_group_permutation_test,
    _isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    _timeseries_correlation_permutation_test,
    two_sample_permutation_test,
]


def call_with(func, *, progress_bar):
    """Invoke `func` with a minimal valid payload and the given progress_bar."""
    rng = np.random.default_rng(0)
    kwargs = {"progress_bar": progress_bar, "random_state": 0}

    if func is one_sample_permutation_test:
        return func(rng.standard_normal((20, 5)), n_permute=20, **kwargs)
    if func is two_sample_permutation_test:
        return func(
            rng.standard_normal(20), rng.standard_normal(20), n_permute=20, **kwargs
        )
    if func is correlation_permutation_test:
        return func(
            rng.standard_normal(30), rng.standard_normal(30), n_permute=20, **kwargs
        )
    if func is matrix_permutation_test:
        a = rng.standard_normal((10, 10))
        b = rng.standard_normal((10, 10))
        a, b = a + a.T, b + b.T
        np.fill_diagonal(a, 0)
        np.fill_diagonal(b, 0)
        return func(a, b, n_permute=20, **kwargs)
    if func is _timeseries_correlation_permutation_test:
        return func(
            rng.standard_normal(60), rng.standard_normal(60), n_permute=20, **kwargs
        )
    if func is _isc_permutation_test:
        return func(rng.standard_normal((20, 20)), n_permute=20, **kwargs)
    if func is _isc_group_permutation_test:
        # Two groups of (n_observations, n_subjects) data.
        return func(
            rng.standard_normal((20, 8)),
            rng.standard_normal((20, 8)),
            n_permute=20,
            **kwargs,
        )
    raise AssertionError(f"no payload defined for {func.__name__}")


@pytest.mark.parametrize("func", PROGRESS_BAR_FUNCTIONS, ids=lambda f: f.__name__)
def test_accepts_progress_bar_kwarg(func):
    """Every inference entry point exposes progress_bar."""
    assert "progress_bar" in inspect.signature(func).parameters


@pytest.mark.parametrize("func", PROGRESS_BAR_FUNCTIONS, ids=lambda f: f.__name__)
def test_progress_bar_defaults_to_false(func):
    """The canonical default is False -- silent unless asked."""
    assert inspect.signature(func).parameters["progress_bar"].default is False


@pytest.mark.parametrize("func", PROGRESS_BAR_FUNCTIONS, ids=lambda f: f.__name__)
def test_silent_by_default(func):
    """No stderr output when progress_bar is left at its default."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        call_with(func, progress_bar=False)
    assert stderr.getvalue() == "", (
        f"{func.__name__} wrote to stderr with progress_bar=False:\n"
        f"{stderr.getvalue()[:200]}"
    )


@pytest.mark.parametrize("func", PROGRESS_BAR_FUNCTIONS, ids=lambda f: f.__name__)
def test_progress_bar_true_emits_output(func):
    """progress_bar=True still shows a bar, so the knob is a real toggle."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        call_with(func, progress_bar=True)
    assert stderr.getvalue() != "", (
        f"{func.__name__} showed no bar with progress_bar=True"
    )
