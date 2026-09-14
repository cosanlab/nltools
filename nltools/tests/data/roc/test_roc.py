import inspect
import warnings

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


def test_nonfinite_scores_are_rejected():
    """A NaN score makes every comparison false, so it is refused up front."""
    with pytest.raises(ValueError, match="finite"):
        Roc(input_values=[np.nan, 0.0, 1.0], binary_outcome=[True, False, True])


def test_replacement_labels_must_contain_both_classes():
    """calculate's binary_outcome= is checked the way the constructor checks it."""
    roc = Roc(input_values=[0.0, 1.0], binary_outcome=[False, True])

    with pytest.raises(ValueError, match="both positive and negative"):
        roc.calculate(binary_outcome=[True, True])


def test_column_shaped_scores_are_treated_as_one_per_observation():
    """An (n, 1) score column is squeezed instead of broadcasting into a matrix."""
    roc = Roc(input_values=np.array([[0.0], [1.0]]), binary_outcome=[False, True])
    roc.calculate()

    assert roc.accuracy == 1.0
    assert roc.misclass.shape == (2,)


def test_integer_scores_score_like_their_float_equivalent():
    """Forced-choice centering cannot be truncated by an integer score dtype."""
    integer = Roc(
        input_values=[2, 1],
        binary_outcome=[True, False],
        forced_choice=np.array([0, 0]),
    )
    integer.calculate()
    floating = Roc(
        input_values=[2.0, 1.0],
        binary_outcome=[True, False],
        forced_choice=np.array([0, 0]),
    )
    floating.calculate()

    assert integer.accuracy == floating.accuracy == 1.0


def test_invalid_method_raises_and_leaves_results_alone():
    """calculate validates method= before it writes any result attribute."""
    input_values, binary_outcome = _make_roc_data()
    roc = Roc(input_values=input_values, binary_outcome=binary_outcome)
    roc.calculate()
    before = (roc.class_thr, roc.accuracy)

    with pytest.raises(ValueError, match="method"):
        roc.calculate(method="not_a_real_method")

    assert (roc.class_thr, roc.accuracy) == before
    fresh = Roc(input_values=input_values, binary_outcome=binary_outcome)
    with pytest.raises(ValueError, match="method"):
        fresh.calculate(method="not_a_real_method")


def test_forced_choice_accepts_a_list_of_string_ids():
    """Subject ids are coerced, so a list of strings pairs up like an array."""
    roc = Roc(
        input_values=[2.0, 1.0],
        binary_outcome=[True, False],
        forced_choice=["a", "a"],
    )
    roc.calculate()

    assert roc.accuracy == 1.0


def test_forced_choice_subject_without_a_pair_is_named():
    """Ids that do not give each subject one positive and one negative are refused."""
    with pytest.raises(ValueError, match="one positive and one negative"):
        Roc(
            input_values=[2.0, 1.0, 0.0, 3.0],
            binary_outcome=[True, True, False, False],
            forced_choice=[0, 0, 1, 1],
        ).calculate()


def test_list_criterion_values_match_the_array_equivalent():
    """criterion_values= is coerced, so a list evaluates the same curve."""
    from_list = Roc(input_values=[0.0, 1.0], binary_outcome=[False, True])
    from_list.calculate(criterion_values=[0.0, 0.5, 2.0])
    from_array = Roc(input_values=[0.0, 1.0], binary_outcome=[False, True])
    from_array.calculate(criterion_values=np.array([0.0, 0.5, 2.0]))

    np.testing.assert_array_equal(from_list.tpr, from_array.tpr)
    np.testing.assert_array_equal(from_list.fpr, from_array.fpr)
    assert from_list.class_thr == from_array.class_thr


def test_default_criterion_values_are_the_empirical_operating_points():
    """The curve is evaluated where the data can move it, not on a fixed grid."""
    tied = Roc(input_values=[1.0, 1.0], binary_outcome=[True, False])
    tied.calculate()
    assert tied.auc == 0.5

    # Perfectly ordered scores, one gap far narrower than any fixed grid step
    ordered = Roc(input_values=[1.0, 1.01, 1000.0], binary_outcome=[False, True, True])
    ordered.calculate()
    assert ordered.auc == 1.0
    assert ordered.accuracy == 1.0


def test_forced_choice_accuracy_does_not_depend_on_row_order():
    """A pair's two errors are matched by subject, not by position."""
    scores = [2.0, 0.0, 1.0, 1.0]
    labels = [True, True, False, False]
    ids = [0, 1, 0, 1]
    order = [0, 1, 3, 2]

    original = Roc(input_values=scores, binary_outcome=labels, forced_choice=ids)
    original.calculate()
    permuted = Roc(
        input_values=[scores[i] for i in order],
        binary_outcome=[labels[i] for i in order],
        forced_choice=[ids[i] for i in order],
    )
    permuted.calculate()

    # Subject 0 ranks its positive above its negative; subject 1 does not
    for roc in (original, permuted):
        assert roc.accuracy == 0.5
        assert roc.n == 2
        assert roc.accuracy_p.k == 1


def test_forced_choice_results_repeat_and_match_pre_centered_scores():
    """Pair centering is derived per call, not written back over the scores."""
    scores = [10.0, 11.0, 20.0, 19.0]
    labels = [True, False, True, False]
    ids = [0, 0, 1, 1]

    roc = Roc(input_values=scores, binary_outcome=labels, forced_choice=ids)
    roc.calculate()
    first = (roc.auc, roc.accuracy)
    roc.calculate()
    assert (roc.auc, roc.accuracy) == first

    pre_centered = Roc(
        input_values=[-0.5, 0.5, 0.5, -0.5],
        binary_outcome=labels,
        forced_choice=ids,
    )
    pre_centered.calculate()
    assert (pre_centered.auc, pre_centered.accuracy) == first
    # The caller's scores are left as they were passed
    assert roc.input_values.tolist() == scores


def test_threshold_is_never_the_corner_above_every_score():
    """With negatives in the majority the "call nothing positive" corner wins the
    count, but a threshold no observation reaches leaves `ppv` undefined."""
    roc = Roc(input_values=[0.0, 1.0, 2.0], binary_outcome=[True, False, False])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        roc.calculate()

    assert np.isfinite(roc.class_thr)
    assert np.isfinite(roc.ppv)


def test_row_shaped_scores_are_rejected():
    """Only trailing singleton axes are squeezed, so a (1, n) row is reported."""
    with pytest.raises(ValueError, match="1-D"):
        Roc(input_values=np.array([[0.0, 1.0]]), binary_outcome=[False, True])


def test_missing_input_values_says_so():
    """A missing input_values is named rather than read as a NaN score."""
    with pytest.raises(ValueError, match="input_values is required"):
        Roc(binary_outcome=[False, True])


def test_criterion_values_without_a_finite_threshold_are_rejected():
    """Thresholds no observation can cross leave every metric undefined."""
    roc = Roc(input_values=[0.0, 1.0], binary_outcome=[False, True])

    with pytest.raises(ValueError, match="at least one finite threshold"):
        roc.calculate(criterion_values=[np.inf, np.inf])
