import inspect

import numpy as np
import pytest

from nltools.data.roc import Roc


def _make_roc_data(seed=0, n=40):
    """Two separable Gaussian classes for a deterministic, non-perfect ROC."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 1.0, n)
    neg = rng.normal(-1.0, 1.0, n)
    input_values = np.concatenate([pos, neg])
    binary_outcome = np.array([True] * n + [False] * n)
    return input_values, binary_outcome


def test_accuracy_se_is_proportion_se():
    """F088: accuracy_se must be sqrt(p*(1-p)/n), not sqrt(p*p/n)."""
    input_values, binary_outcome = _make_roc_data()
    roc = Roc(input_values=input_values, binary_outcome=binary_outcome)
    roc.calculate()

    p = np.mean(~roc.misclass)
    expected = np.sqrt(p * (1 - p) / roc.n)
    buggy = p / np.sqrt(roc.n)  # sqrt(p*p/n)

    assert np.isclose(roc.accuracy_se, expected)
    # Guard: on a non-perfect classifier the two formulas disagree, so this
    # test genuinely exercises the fix.
    assert not np.isclose(roc.accuracy_se, buggy)


def test_calculate_accepts_python_list_inputs():
    """F090: calculate() must coerce list binary_outcome/input_values like __init__."""
    input_values, binary_outcome = _make_roc_data()
    roc = Roc(input_values=input_values, binary_outcome=binary_outcome)

    # Passing plain Python lists to calculate() used to break `~self.binary_outcome`
    # (TypeError on ~list) and `self.input_values.squeeze()` (list has no squeeze).
    roc.calculate(
        input_values=list(input_values),
        binary_outcome=[bool(x) for x in binary_outcome],
    )

    assert isinstance(roc.binary_outcome, np.ndarray)
    assert roc.binary_outcome.dtype == bool
    assert isinstance(roc.input_values, np.ndarray)
    assert np.isfinite(roc.accuracy)


def test_method_is_canonical_variant_kwarg():
    """F097/F195: `method=` selects the threshold variant on __init__/calculate.

    The old `threshold_type=` spelling is gone across the release.
    """
    input_values, binary_outcome = _make_roc_data()

    # __init__ accepts method= and stores it on self.method
    roc = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )
    assert roc.method == "optimal_balanced"

    # calculate accepts method= and updates self.method
    roc.calculate(method="optimal_overall")
    assert roc.method == "optimal_overall"

    # The banned aliases are gone from both signatures.
    init_params = inspect.signature(Roc.__init__).parameters
    calc_params = inspect.signature(Roc.calculate).parameters
    assert "threshold_type" not in init_params
    assert "threshold_type" not in calc_params
    assert "method" in init_params
    assert "method" in calc_params

    # Bad method value still raises in __init__.
    with pytest.raises(ValueError):
        Roc(
            input_values=input_values,
            binary_outcome=binary_outcome,
            method="not_a_real_method",
        )


def test_plot_method_kwarg_is_method():
    """F097/F195: Roc.plot selects its variant via `method=`, not `plot_method=`."""
    plot_params = inspect.signature(Roc.plot).parameters
    assert "method" in plot_params
    assert "plot_method" not in plot_params


def test_roc_signatures_reject_stray_kwargs():
    """F096: no dead **kwargs on __init__ or plot to silently swallow typos."""
    input_values, binary_outcome = _make_roc_data()

    init_params = inspect.signature(Roc.__init__).parameters
    plot_params = inspect.signature(Roc.plot).parameters
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in init_params.values()
    )
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in plot_params.values()
    )

    # A typo'd kwarg is now a hard error rather than being swallowed.
    with pytest.raises(TypeError):
        Roc(
            input_values=input_values,
            binary_outcome=binary_outcome,
            typoed_kwarg=True,
        )
