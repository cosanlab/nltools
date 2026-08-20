"""
API-convention guards for the inference layer.

`nltools/algorithms/inference` is public in practice: it is documented in
`docs/api/algorithms/inference.md`, re-exported through `nltools.stats`, and
imported directly by downstream code. These tests hold it to the conventions in
CLAUDE.md rather than treating it as private internals.

They exist because a real bug got through: inserting a parameter into a helper
signature ahead of an existing one silently shifted a positional argument at the
dispatch site, turning a progress bar back on with no error.
"""

import inspect

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

# Names exported from BOTH nltools.stats and nltools.algorithms.inference as
# distinct function objects. The facade deliberately renames the backend-
# selection kwarg; nothing else may differ.
TRANSLATED_KWARGS = {"device", "parallel", "backend"}

PAIRED_FUNCTION_NAMES = [
    "correlation_permutation_test",
    "matrix_permutation_test",
    "one_sample_permutation_test",
    "timeseries_correlation_permutation_test",
    "two_sample_permutation_test",
]


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


@pytest.mark.parametrize("name", PAIRED_FUNCTION_NAMES)
def test_facade_and_engine_signatures_agree(name):
    """The nltools.stats wrapper and the engine must not drift apart.

    They are separate function objects with the same name, so a change applied
    to one silently leaves the other behind. Only the documented backend-kwarg
    rename (parallel/backend -> device) may differ.
    """
    import nltools.algorithms.inference as engine_mod

    facade = getattr(stats_facade, name)
    engine = getattr(engine_mod, name)
    assert facade is not engine, f"{name}: expected two distinct objects"

    facade_params = inspect.signature(facade).parameters
    engine_params = inspect.signature(engine).parameters

    facade_names = set(facade_params) - TRANSLATED_KWARGS
    engine_names = set(engine_params) - TRANSLATED_KWARGS

    missing = engine_names - facade_names
    extra = facade_names - engine_names
    assert not missing, (
        f"nltools.stats.{name} is missing engine params: {sorted(missing)}"
    )
    assert not extra, (
        f"nltools.stats.{name} has params the engine lacks: {sorted(extra)}"
    )

    drifted = {
        k: (facade_params[k].default, engine_params[k].default)
        for k in facade_names & engine_names
        if facade_params[k].default is not inspect.Parameter.empty
        and engine_params[k].default is not inspect.Parameter.empty
        and facade_params[k].default != engine_params[k].default
    }
    assert not drifted, f"nltools.stats.{name} default drift vs engine: {drifted}"


# NOTE: `isc` is a function in nltools.stats but a module in
# nltools.algorithms.inference, and the two layers export 13 same-named,
# distinct objects overall. That duplication is a design question tracked in
# issue #474, not something this PR resolves -- the parity test above is the
# interim guard that keeps the pairs from drifting while it is decided.
