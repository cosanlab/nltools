"""
API-convention guards for the inference layer.

`nltools/algorithms/inference` is public in practice: it is documented in
`docs/api/algorithms/inference.md`, re-exported through `nltools.algorithms`, and
imported directly by downstream code. These tests hold it to the conventions in
CLAUDE.md rather than treating it as private internals.

They exist because a real bug got through: inserting a parameter into a helper
signature ahead of an existing one silently shifted a positional argument at the
dispatch site, turning a progress bar back on with no error.
"""

import importlib
import inspect

import pytest

import nltools.algorithms as algorithms

from nltools.algorithms.inference import (
    correlation_permutation_test,
    isc_group_permutation_test,
    isc_permutation_test,
    matrix_permutation_test,
    one_sample_permutation_test,
    timeseries_correlation_permutation_test,
    two_sample_permutation_test,
)
from nltools.algorithms.inference.intersubject import isc, isc_group, isfc, isps

# Public entry points, mapped to the leading data arguments that may legitimately
# be passed positionally. Everything after them must be keyword-only.
PUBLIC_ENTRY_POINTS = {
    one_sample_permutation_test: ["data"],
    two_sample_permutation_test: ["data1", "data2"],
    correlation_permutation_test: ["data1", "data2"],
    matrix_permutation_test: ["data1", "data2"],
    timeseries_correlation_permutation_test: ["data1", "data2"],
    isc_permutation_test: ["data"],
    isc_group_permutation_test: ["group1", "group2"],
}

# The user-facing ISC family (`nltools.algorithms.inference.intersubject`), held
# to the same keyword-only convention as a set rather than function by function —
# `isfc` was the one holdout (q31x 2p68).
ISC_FAMILY_ENTRY_POINTS = {
    isc: ["data"],
    isc_group: ["group1", "group2"],
    isfc: ["data"],
    isps: ["data"],
}


@pytest.mark.parametrize(
    "func,data_args",
    PUBLIC_ENTRY_POINTS.items(),
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_options_are_keyword_only(func, data_args):
    """Everything after the data arguments must be keyword-only.

    CLAUDE.md requires a `*` marker on any public function with 3+ kwargs. It is
    also what makes a parameter insertion a TypeError instead of a silent
    value shift at the call site.
    """
    params = list(inspect.signature(func).parameters.values())
    positional = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    offenders = [p.name for p in positional if p.name not in data_args]
    assert not offenders, (
        f"{func.__name__} accepts {offenders} positionally; "
        f"only {data_args} may be positional. Add a `*` marker."
    )


@pytest.mark.parametrize(
    "func,data_args",
    ISC_FAMILY_ENTRY_POINTS.items(),
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_isc_family_options_are_keyword_only(func, data_args):
    """Everything after the data arguments must be keyword-only, across the family.

    `isfc` was the ISC-family holdout: it took `method` and `n_jobs`
    positionally with no `random_state`/`progress_bar` at all. Checking the
    family as one parametrized set (rather than a one-off test per function)
    is what keeps a future addition like `isfc` from going unchecked again.
    """
    params = list(inspect.signature(func).parameters.values())
    positional = [p for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
    offenders = [p.name for p in positional if p.name not in data_args]
    assert not offenders, (
        f"{func.__name__} accepts {offenders} positionally; "
        f"only {data_args} may be positional. Add a `*` marker."
    )


def test_matrix_trailing_kwarg_order():
    """CLAUDE.md trailing order: ..., return flags, n_jobs, random_state, progress_bar.

    `matrix_permutation_test` drifted (`progress_bar` landed before
    `random_state`, `return_null` after `n_jobs`); all params are keyword-only,
    so pinning the canonical order is not a breaking change.
    """
    params = list(inspect.signature(matrix_permutation_test).parameters)
    assert params == [
        "data1",
        "data2",
        "n_permute",
        "metric",
        "how",
        "include_diag",
        "tail",
        "return_null",
        "device",
        "n_jobs",
        "random_state",
        "progress_bar",
    ]


# The permutation entry points re-exported by `nltools.algorithms` must be the
# engine functions themselves: the v0.6.0 consolidation (issue #474) removed the
# wrapper layer, so facade/engine drift is structurally impossible only as long
# as the identity below holds.
ENGINE_IDENTITY = {
    "one_sample_permutation_test": "nltools.algorithms.inference.one_sample",
    "two_sample_permutation_test": "nltools.algorithms.inference.two_sample",
    "correlation_permutation_test": "nltools.algorithms.inference.correlation",
    "timeseries_correlation_permutation_test": "nltools.algorithms.inference.timeseries",
    "circle_shift": "nltools.algorithms.inference.timeseries",
    "phase_randomize": "nltools.algorithms.inference.timeseries",
    "matrix_permutation_test": "nltools.algorithms.inference.matrix",
    "double_center": "nltools.algorithms.inference.matrix",
    "u_center": "nltools.algorithms.inference.matrix",
    "distance_correlation": "nltools.algorithms.inference.matrix",
}


@pytest.mark.parametrize("name", sorted(ENGINE_IDENTITY))
def test_permutation_functions_are_engine_functions(name):
    engine_mod = importlib.import_module(ENGINE_IDENTITY[name])
    assert getattr(algorithms, name) is getattr(engine_mod, name), (
        f"algorithms.{name} is a wrapper, not the engine function — "
        "the v0.6.0 consolidation removed the wrapper layer"
    )
