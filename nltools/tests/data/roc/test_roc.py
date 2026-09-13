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

    # calculate accepts an explicit method= override for that call only; it
    # does not change the instance's configured method (q31x fvgk #12).
    roc.calculate(method="optimal_overall")
    assert roc.method == "optimal_balanced"

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


def _make_imbalanced_roc_data(seed=0, n_positive=40, n_negative=10):
    """Two separable Gaussian classes of unequal size.

    Weighting the classes equally and weighting the observations equally pick
    different thresholds here, which is what makes them distinguishable.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 1.0, n_positive)
    neg = rng.normal(-1.0, 1.0, n_negative)
    input_values = np.concatenate([pos, neg])
    binary_outcome = np.array([True] * n_positive + [False] * n_negative)
    return input_values, binary_outcome


def _make_forced_choice_data(seed=1, n_subjects=15):
    """Paired positive/negative decision values, one subject per index."""
    rng = np.random.default_rng(seed)
    pos = rng.normal(1.0, 1.0, n_subjects)
    neg = rng.normal(-1.0, 1.0, n_subjects)
    input_values = np.concatenate([pos, neg])
    binary_outcome = np.array([True] * n_subjects + [False] * n_subjects)
    forced_choice = np.concatenate([np.arange(n_subjects), np.arange(n_subjects)])
    return input_values, binary_outcome, forced_choice


def test_calculate_defaults_to_constructor_method():
    """q31x fvgk (#12): calculate() with no method= uses the constructor's method."""
    # Imbalanced classes, so the balanced and overall rules genuinely disagree:
    # with equal class sizes they pick the same threshold and the guard below
    # would pass for the wrong reason.
    input_values, binary_outcome = _make_imbalanced_roc_data()

    roc = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )
    roc.calculate()
    assert roc.method == "optimal_balanced"

    matching = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )
    matching.calculate(method="optimal_balanced")
    assert roc.class_thr == matching.class_thr

    # Guard: on this dataset 'optimal_overall' genuinely picks a different
    # threshold, so a silent fallback to it would be caught here.
    mismatching = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )
    mismatching.calculate(method="optimal_overall")
    assert roc.class_thr != mismatching.class_thr


def test_calculate_explicit_method_overrides_for_one_call():
    """q31x fvgk (#12): an explicit calculate(method=...) is a one-call override."""
    input_values, binary_outcome = _make_roc_data()
    roc = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )

    roc.calculate(method="minimum_sdt_bias")
    reference = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="minimum_sdt_bias",
    )
    reference.calculate(method="minimum_sdt_bias")
    assert roc.class_thr == reference.class_thr
    # The override does not stick past the call it was passed to.
    assert roc.method == "optimal_balanced"

    # A subsequent bare calculate() reverts to the constructor's method.
    roc.calculate()
    balanced_reference = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )
    balanced_reference.calculate(method="optimal_balanced")
    assert roc.class_thr == balanced_reference.class_thr


def test_plot_does_not_mutate_calculate_results():
    """q31x fvgk (#13): plot() must not overwrite calculate()'s stored results."""
    input_values, binary_outcome, forced_choice = _make_forced_choice_data()
    roc = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        forced_choice=forced_choice,
        method="optimal_balanced",
    )
    roc.calculate()
    before = (roc.sensitivity, roc.specificity, roc.ppv, roc.auc, roc.method)

    roc.plot(method="gaussian")

    after = (roc.sensitivity, roc.specificity, roc.ppv, roc.auc, roc.method)
    assert before == after

    # The Gaussian-model estimates plot() draws land on their own attributes.
    assert hasattr(roc, "gaussian_sensitivity")
    assert hasattr(roc, "gaussian_specificity")
    assert hasattr(roc, "gaussian_ppv")
    assert hasattr(roc, "gaussian_auc")


def test_optimal_balanced_maximizes_balanced_accuracy():
    """`optimal_balanced` weights the two classes equally.

    Maximizing `(tpr + fpr) / 2` is maximized by calling every observation
    positive, so the rule always collapsed to the lowest criterion value:
    perfect sensitivity, zero specificity. Balanced accuracy is
    `(tpr + (1 - fpr)) / 2`.
    """
    input_values, binary_outcome = _make_roc_data()
    roc = Roc(
        input_values=input_values,
        binary_outcome=binary_outcome,
        method="optimal_balanced",
    )
    roc.calculate()

    # The behavioural guard first: the old rule scored specificity 0.00 on every
    # dataset, so this pair is what was red.
    assert roc.sensitivity > 0.5
    assert roc.specificity > 0.5
    # Then the rule itself, so a threshold that merely happens to be decent
    # cannot pass for the argmax of balanced accuracy.
    balanced = (roc.tpr + (1 - roc.fpr)) / 2
    assert roc.class_thr == roc.criterion_values[np.argmax(balanced)]
