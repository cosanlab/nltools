import numpy as np

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
