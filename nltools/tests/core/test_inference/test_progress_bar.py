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

import nltools.stats as stats_facade
from nltools.algorithms.inference import (
    correlation_permutation_test,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)

# Every algorithm-layer entry point that drives a tqdm loop.
PROGRESS_BAR_FUNCTIONS = [
    correlation_permutation_test,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
]

# `nltools.stats` re-exports thin wrappers (they translate parallel= -> device=),
# so they are *different function objects* from the algorithm-layer ones above
# and need the knob independently. These are what users actually import.
FACADE_FUNCTION_NAMES = [
    "correlation_permutation_test",
    "matrix_permutation_test",
    "one_sample_permutation_test",
    "timeseries_correlation_permutation_test",
    "two_sample_permutation_test",
]


def call_with(func, *, progress_bar, facade_name=None):
    """Invoke `func` with a minimal valid payload and the given progress_bar.

    `facade_name` selects the payload when `func` is a `nltools.stats` wrapper
    rather than one of the algorithm-layer function objects.
    """
    rng = np.random.default_rng(0)
    kwargs = {"progress_bar": progress_bar, "random_state": 0}
    key = facade_name

    if key == "one_sample_permutation_test":
        return func(rng.standard_normal((20, 5)), n_permute=20, **kwargs)
    if key == "two_sample_permutation_test":
        return func(
            rng.standard_normal(20), rng.standard_normal(20), n_permute=20, **kwargs
        )
    if key == "correlation_permutation_test":
        return func(
            rng.standard_normal(30), rng.standard_normal(30), n_permute=20, **kwargs
        )
    if key == "timeseries_correlation_permutation_test":
        return func(
            rng.standard_normal(60), rng.standard_normal(60), n_permute=20, **kwargs
        )
    if key == "matrix_permutation_test":
        a = rng.standard_normal((10, 10))
        b = rng.standard_normal((10, 10))
        a, b = a + a.T, b + b.T
        np.fill_diagonal(a, 0)
        np.fill_diagonal(b, 0)
        return func(a, b, n_permute=20, **kwargs)

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
    if func is timeseries_correlation_permutation_test:
        return func(
            rng.standard_normal(60), rng.standard_normal(60), n_permute=20, **kwargs
        )
    if func is isc_permutation_test:
        return func(rng.standard_normal((20, 20)), n_permute=20, **kwargs)
    if func is isc_group_permutation_test:
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


@pytest.mark.parametrize("name", FACADE_FUNCTION_NAMES)
def test_stats_facade_exposes_progress_bar(name):
    """`nltools.stats` wrappers expose the knob too, defaulting to False."""
    func = getattr(stats_facade, name)
    param = inspect.signature(func).parameters.get("progress_bar")
    assert param is not None, f"nltools.stats.{name} is missing progress_bar"
    assert param.default is False


@pytest.mark.parametrize("name", FACADE_FUNCTION_NAMES)
def test_stats_facade_silent_by_default(name):
    """The wrappers must actually forward the default, not just accept it."""
    func = getattr(stats_facade, name)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        call_with(func, progress_bar=False, facade_name=name)
    assert stderr.getvalue() == "", (
        f"nltools.stats.{name} wrote to stderr by default:\n{stderr.getvalue()[:200]}"
    )
