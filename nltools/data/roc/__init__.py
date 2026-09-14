"""ROC (Receiver Operating Characteristic) analysis for single-interval classification.

These tools provide the ability to quickly run receiver operating characteristic
analyses on the output of machine-learning models applied to imaging data.
"""

import numpy as np
from nltools.plotting import _plot_roc
from scipy.stats import norm, binomtest
from sklearn.metrics import auc

_VALID_METHODS = ("optimal_overall", "optimal_balanced", "minimum_sdt_bias")


def _squeezed_trailing(values):
    """Drop trailing singleton axes, leaving any leading axis in place.

    An `(n, 1)` column of one value per observation becomes `(n,)`; a `(1, n)`
    row stays two-dimensional, so it is reported rather than silently read as n
    observations.

    Args:
        values (np.ndarray): Array to squeeze.

    Returns:
        np.ndarray: The array with its trailing singleton axes dropped.
    """
    while values.ndim > 1 and values.shape[-1] == 1:
        values = values[..., 0]
    return np.atleast_1d(values)


def _validated_method(method):
    """Check a threshold-selection variant name.

    Args:
        method (str): Variant name to check.

    Returns:
        str: The same name.

    Raises:
        ValueError: If `method` is not one of the three variants.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method must be one of {list(_VALID_METHODS)}, got {method!r}"
        )
    return method


def _validated_scores(input_values):
    """Coerce decision values to an owned 1-D float array.

    Args:
        input_values (array-like): Decision values, one per observation.

    Returns:
        np.ndarray: 1-D float array.

    Raises:
        ValueError: If the values are not 1-D once trailing singleton axes are
            dropped, or if any of them is not finite.
    """
    scores = _squeezed_trailing(np.asarray(input_values, dtype=float))
    if scores.ndim != 1:
        raise ValueError(
            "input_values must be 1-D, one decision value per observation; got "
            f"shape {np.shape(input_values)}."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "input_values must all be finite; NaN or infinite decision values "
            "make every threshold comparison false and the metrics meaningless."
        )
    return scores


def _validated_outcome(binary_outcome):
    """Coerce class labels to an owned 1-D boolean array.

    Args:
        binary_outcome (array-like): Class label per observation.

    Returns:
        np.ndarray: 1-D boolean array.

    Raises:
        ValueError: If the labels are not 1-D, or if one class is missing.
    """
    labels = _squeezed_trailing(np.asarray(binary_outcome)).astype(bool)
    if labels.ndim != 1:
        raise ValueError(
            "binary_outcome must be 1-D, one label per observation; got "
            f"shape {np.shape(binary_outcome)}."
        )
    if labels.all() or not labels.any():
        raise ValueError(
            "binary_outcome must contain both positive and negative cases "
            "(True and False)."
        )
    return labels


def _validated_subject_ids(forced_choice, n_observations):
    """Coerce forced-choice subject ids to an owned 1-D array.

    Args:
        forced_choice (array-like): Subject id per observation.
        n_observations (int): Number of observations the ids must cover.

    Returns:
        np.ndarray: 1-D array of subject ids.

    Raises:
        ValueError: If the ids are not 1-D or do not cover every observation.
    """
    ids = _squeezed_trailing(np.asarray(forced_choice))
    if ids.ndim != 1:
        raise ValueError(
            "forced_choice must be 1-D, one subject id per observation; got "
            f"shape {np.shape(forced_choice)}."
        )
    if len(ids) != n_observations:
        raise ValueError(
            f"forced_choice has {len(ids)} subject ids for {n_observations} "
            "observations; it needs one id per observation."
        )
    return ids


def _validated_criterion_values(criterion_values):
    """Coerce caller-supplied thresholds to an owned 1-D float array.

    Args:
        criterion_values (array-like): Thresholds to evaluate the curve at.

    Returns:
        np.ndarray: 1-D float array.

    Raises:
        ValueError: If the thresholds are not 1-D.
    """
    values = _squeezed_trailing(np.asarray(criterion_values, dtype=float))
    if values.ndim != 1:
        raise ValueError(
            f"criterion_values must be 1-D; got shape {np.shape(criterion_values)}."
        )
    return values


def _validated_inputs(input_values, binary_outcome, forced_choice):
    """Coerce and cross-check the three per-observation inputs together.

    Args:
        input_values (array-like): Decision values.
        binary_outcome (array-like): Class labels.
        forced_choice (array-like | None): Subject ids, or None for
            single-interval classification.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray | None]: Scores, labels and
            subject ids, each owned by the caller.

    Raises:
        ValueError: If `input_values` is missing, if any input fails its own
            checks, or if the scores and the labels are different lengths.
    """
    if input_values is None:
        raise ValueError("input_values is required.")
    scores = _validated_scores(input_values)
    labels = _validated_outcome(binary_outcome)
    if len(scores) != len(labels):
        raise ValueError(
            f"input_values has {len(scores)} values and binary_outcome has "
            f"{len(labels)} labels; they must be the same length."
        )
    ids = (
        None
        if forced_choice is None
        else _validated_subject_ids(forced_choice, len(labels))
    )
    return scores, labels, ids


def _forced_choice_pairs(forced_choice, binary_outcome):
    """Locate each subject's positive and negative observation.

    Args:
        forced_choice (np.ndarray): Subject id per observation.
        binary_outcome (np.ndarray): Boolean label per observation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Row index of the positive observation and
            of the negative observation, both in the same subject order.

    Raises:
        ValueError: If a subject does not contribute exactly one positive and
            one negative observation.
    """
    positive_idx = []
    negative_idx = []
    for subject in np.unique(forced_choice):
        in_subject = forced_choice == subject
        positives = np.flatnonzero(in_subject & binary_outcome)
        negatives = np.flatnonzero(in_subject & ~binary_outcome)
        if len(positives) != 1 or len(negatives) != 1:
            raise ValueError(
                f"forced_choice subject {subject!r} has {len(positives)} positive "
                f"and {len(negatives)} negative observations; each subject must "
                "contribute exactly one positive and one negative observation."
            )
        positive_idx.append(positives[0])
        negative_idx.append(negatives[0])
    return np.array(positive_idx, dtype=int), np.array(negative_idx, dtype=int)


def _centered_within_pairs(scores, positive_idx, negative_idx):
    """Center each subject's two scores on their own mean.

    Args:
        scores (np.ndarray): 1-D decision values.
        positive_idx (np.ndarray): Row index of each subject's positive observation.
        negative_idx (np.ndarray): Row index of each subject's negative observation.

    Returns:
        np.ndarray: A new array holding the centered values.
    """
    centered = scores.copy()
    pair_means = (scores[positive_idx] + scores[negative_idx]) / 2
    centered[positive_idx] = scores[positive_idx] - pair_means
    centered[negative_idx] = scores[negative_idx] - pair_means
    return centered


def _default_criterion_values(scores):
    """Operating points of the empirical ROC curve for `scores`.

    Classification is `score >= criterion`, so the curve can only move at a value
    some observation actually takes; the extra point above the largest score is
    the corner where nothing is called positive. These are the thresholds
    `sklearn.metrics.roc_curve` evaluates.

    Args:
        scores (np.ndarray): 1-D decision values the curve is evaluated against.

    Returns:
        np.ndarray: Ascending thresholds, ending above the largest score.
    """
    return np.append(np.unique(scores), np.inf)


class Roc:
    """Compute receiver operating characteristic curves for single-interval or forced-choice classification.

    The Roc class is based on Tor Wager's Matlab roc_plot.m function and
    allows a user to easily run different types of receiver operator
    characteristic curves.  For example, one might be interested in single
    interval or forced choice.

    Args:
        input_values (array-like): 1-D continuous decision values, one per observation.
        binary_outcome (array-like): Boolean class label per observation.
        method (str): Threshold-selection variant, naming what the chosen
            threshold maximizes or minimizes: `'optimal_overall'` maximizes the
            number of correct classifications, so the larger class dominates;
            `'optimal_balanced'` maximizes balanced accuracy, the mean of
            sensitivity and specificity, weighting the two classes equally;
            `'minimum_sdt_bias'` minimizes the signal-detection response bias
            `c`, which places the threshold midway between the two classes'
            estimated distributions. With equal class sizes the first two often
            agree.
        forced_choice (array-like, optional): Subject id per observation for
            forced-choice classification (each subject contributes one positive and
            one negative observation).

    Attributes:
        input_values (np.ndarray): Decision values, as given, coerced to a 1-D
            float array. Forced-choice pair centering is derived inside
            `calculate` and never written back here.
        binary_outcome (np.ndarray): Boolean labels.
        method (str): Configured threshold-selection variant. Set at construction;
            `calculate`'s `method=` argument reads this as its default and never
            writes back to it, so an explicit override passed to `calculate` only
            affects that call.
        forced_choice (np.ndarray | None): Subject ids for forced-choice classification.
        criterion_values (np.ndarray): Thresholds at which `tpr`/`fpr` were evaluated;
            set by `calculate`. By default the distinct decision values in
            ascending order, plus one threshold above the largest.
        tpr (np.ndarray): True positive rate per criterion value; set by `calculate`.
        fpr (np.ndarray): False positive rate per criterion value; set by `calculate`.
        auc (float): Area under the ROC curve; set by `calculate`.
        class_thr (float): Selected classification threshold; set by `calculate`.
        sensitivity (float): Sensitivity at `class_thr`; set by `calculate`.
        specificity (float): Specificity at `class_thr`; set by `calculate`.
        ppv (float): Positive predictive value at `class_thr`; set by `calculate`.
        accuracy (float): Classification accuracy; set by `calculate`.
        accuracy_se (float): Standard error of the accuracy; set by `calculate`.
        accuracy_p (BinomTestResult): `scipy.stats.binomtest` result comparing accuracy
            against chance (read `.pvalue`); set by `calculate`.
        tpr_smooth (np.ndarray): Gaussian-model true positive rate curve; set by
            `plot(method='gaussian')`. Never read by `calculate`.
        fpr_smooth (np.ndarray): Gaussian-model false positive rate curve; set by
            `plot(method='gaussian')`. Never read by `calculate`.
        aucn (float): Area under the Gaussian-model curve (`tpr_smooth`/`fpr_smooth`);
            set by `plot(method='gaussian')`. Never read by `calculate`.
        gaussian_sensitivity (float): Gaussian-model sensitivity estimate for
            forced-choice data; set by `plot(method='gaussian')`. Never read by
            `calculate`.
        gaussian_specificity (float): Gaussian-model specificity estimate for
            forced-choice data; set by `plot(method='gaussian')`. Never read by
            `calculate`.
        gaussian_ppv (float): Gaussian-model positive predictive value for
            forced-choice data; set by `plot(method='gaussian')`. Never read by
            `calculate`.
        gaussian_auc (float): Gaussian-model area under the curve for forced-choice
            data; set by `plot(method='gaussian')`. Never read by `calculate`.

    Raises:
        ValueError: If `input_values` is not 1-D or holds a non-finite value, if
            it and `binary_outcome` have different lengths, if `binary_outcome`
            holds only one class, if `forced_choice` does not carry one subject
            id per observation, or if `method` is not one of the three variants.
    """

    def __init__(
        self,
        *,
        input_values=None,
        binary_outcome=None,
        method="optimal_overall",
        forced_choice=None,
    ):
        scores, labels, subject_ids = _validated_inputs(
            input_values, binary_outcome, forced_choice
        )
        self.input_values = scores
        self.binary_outcome = labels
        self.forced_choice = subject_ids
        self.method = _validated_method(method)

    def calculate(
        self,
        *,
        input_values=None,
        binary_outcome=None,
        criterion_values=None,
        method=None,
        forced_choice=None,
        balanced_acc=False,
        tail=2,
    ):
        """Calculate ROC metrics and store them on the instance.

        Args:
            input_values (array-like, optional): 1-D continuous decision values, one
                per observation. Defaults to the values given at construction.
            binary_outcome (array-like, optional): Boolean class label per
                observation. Defaults to the labels given at construction.
            criterion_values (array-like, optional): Thresholds at which to evaluate
                `fpr` and `tpr`. Defaults to the empirical operating points: each
                distinct decision value in ascending order, plus one threshold
                above the largest, where nothing is called positive.
            method (str, optional): Threshold-selection variant, one of
                `'optimal_overall'` (maximize correct classifications),
                `'optimal_balanced'` (maximize balanced accuracy, the mean of
                sensitivity and specificity), or `'minimum_sdt_bias'` (minimize
                signal-detection response bias).
                Defaults to `None`, which uses the instance's configured `method`
                (set at construction, or by assigning `self.method` directly). An
                explicit value overrides the configured `method` for this call only
                and does not change `self.method`.
            forced_choice (array-like, optional): Subject id per observation for
                forced-choice classification.
            balanced_acc (bool): Report balanced accuracy (mean of sensitivity and
                specificity) instead of overall accuracy. Only affects the accuracy
                estimate, not the p-value or the threshold used for
                sensitivity/specificity.
            tail (int | str): `2`/`'two'` for two-tailed (default); `1`/`'one'` for
                one-tailed (accuracy > chance) in the binomial test for `accuracy_p`.

        Raises:
            ValueError: If any replacement input fails the checks the constructor
                applies, if `method` is not one of the three variants, or if a
                `forced_choice` subject does not contribute exactly one positive
                and one negative observation. Nothing is written when it raises.
        """
        from nltools.algorithms.validation import _validate_tail_parameter

        binom_alternative = (
            "two-sided" if _validate_tail_parameter(tail) == "two" else "greater"
        )

        # An explicit method= overrides the instance's configured self.method for
        # this call only; self.method itself is left untouched so a later bare
        # calculate() reverts to it (q31x fvgk #12). Resolve it before any result
        # attribute is written, so a bad name leaves the last results intact.
        resolved_method = _validated_method(self.method if method is None else method)

        scores, labels, subject_ids = _validated_inputs(
            self.input_values if input_values is None else input_values,
            self.binary_outcome if binary_outcome is None else binary_outcome,
            self.forced_choice if forced_choice is None else forced_choice,
        )
        self.input_values = scores
        self.binary_outcome = labels
        self.forced_choice = subject_ids

        # Forced choice scores each subject's pair against the pair's own mean.
        # The centered values are derived here rather than written back over the
        # scores the caller passed in, so every call evaluates the same array.
        if subject_ids is None:
            positive_idx = negative_idx = None
            evaluated = scores
        else:
            positive_idx, negative_idx = _forced_choice_pairs(subject_ids, labels)
            evaluated = _centered_within_pairs(scores, positive_idx, negative_idx)

        # Create Criterion Values
        if criterion_values is not None:
            self.criterion_values = _validated_criterion_values(criterion_values)
        else:
            self.criterion_values = _default_criterion_values(evaluated)

        # Calculate true positive and false positive rate
        self.tpr = np.zeros(self.criterion_values.shape)
        self.fpr = np.zeros(self.criterion_values.shape)
        for i, x in enumerate(self.criterion_values):
            wh = evaluated >= x
            self.tpr[i] = np.sum(wh[labels]) / np.sum(labels)
            self.fpr[i] = np.sum(wh[~labels]) / np.sum(~labels)
        self.n_true = np.sum(labels)
        self.n_false = np.sum(~labels)
        self.auc = auc(self.fpr, self.tpr)

        # Get criterion threshold. The corner above the largest score belongs on
        # the curve but not in the selection: a threshold no observation reaches
        # calls nothing positive, which leaves the positive predictive value
        # undefined. Choose among the thresholds the data can actually cross.
        selectable = np.isfinite(self.criterion_values)
        if not selectable.any():
            # Only reachable when the caller supplies no finite threshold
            selectable = np.ones_like(selectable)
        candidates = self.criterion_values[selectable]
        tpr = self.tpr[selectable]
        fpr = self.fpr[selectable]
        if subject_ids is not None:
            # Centering puts a pair's two scores either side of zero
            self.class_thr = 0
        elif resolved_method == "optimal_balanced":
            # Balanced accuracy is the mean of sensitivity and specificity.
            # Averaging tpr with fpr instead maximizes at the lowest
            # criterion value, where everything is called positive.
            balanced_accuracy = (tpr + (1 - fpr)) / 2
            self.class_thr = candidates[np.argmax(balanced_accuracy)]
        elif resolved_method == "optimal_overall":
            n_corr_t = tpr * self.n_true
            n_corr_f = (1 - fpr) * self.n_false
            sm = n_corr_t + n_corr_f
            self.class_thr = candidates[np.argmax(sm)]
        elif resolved_method == "minimum_sdt_bias":
            # Calculate  MacMillan and Creelman 2005 Response Bias (c_bias)
            c_bias = (
                norm.ppf(np.maximum(0.0001, np.minimum(0.9999, tpr)))
                + norm.ppf(np.maximum(0.0001, np.minimum(0.9999, fpr)))
            ) / float(2)
            self.class_thr = candidates[np.argmin(abs(c_bias))]

        # Calculate output
        self.false_positive = (evaluated >= self.class_thr) & (~labels)
        self.false_negative = (evaluated < self.class_thr) & labels
        self.misclass = (self.false_negative) | (self.false_positive)
        self.true_positive = labels & (~self.misclass)
        self.true_negative = (~labels) & (~self.misclass)
        self.sensitivity = np.sum(evaluated[labels] >= self.class_thr) / self.n_true
        self.specificity = (
            1 - np.sum(evaluated[~labels] >= self.class_thr) / self.n_false
        )
        self.ppv = np.sum(self.true_positive) / (
            np.sum(self.true_positive) + np.sum(self.false_positive)
        )
        if subject_ids is not None:
            # One entry per subject, positives and negatives in the same subject
            # order, so the two halves of a pair line up however the rows were
            # ordered.
            self.true_positive = self.true_positive[positive_idx]
            self.true_negative = self.true_negative[negative_idx]
            self.false_negative = self.false_negative[positive_idx]
            self.false_positive = self.false_positive[negative_idx]
            self.misclass = (self.false_positive) | (self.false_negative)

        # Calculate Accuracy
        if balanced_acc:
            self.accuracy = np.mean(
                [self.sensitivity, self.specificity]
            )  # See Brodersen, Ong, Stephan, Buhmann (2010)
        else:
            self.accuracy = 1 - np.mean(self.misclass)

        # Calculate p-Value using binomial test (can add hierarchical version of binomial test)
        self.n = len(self.misclass)
        self.accuracy_p = binomtest(
            int(np.sum(~self.misclass)), self.n, p=0.5, alternative=binom_alternative
        )
        p = np.mean(~self.misclass)
        self.accuracy_se = np.sqrt(p * (1 - p) / self.n)

    def plot(self, *, method="gaussian", balanced_acc=False):
        """Create a ROC plot.

        Runs `calculate` first, then plots either a Gaussian-smoothed ROC curve fit
        to the decision values or the observed empirical curve. The underlying
        `calculate` call re-runs with the instance's configured `method` (it never
        overrides the threshold rule), and the Gaussian-model curve estimates are
        stored on their own attributes rather than overwriting `calculate`'s
        `sensitivity`, `specificity`, `ppv`, and `auc`.

        Args:
            method (str): Type of plot, `'gaussian'` or `'observed'`.
            balanced_acc (bool): Passed to `calculate`; report balanced accuracy.

        Returns:
            matplotlib.figure.Figure: The ROC figure.

        Note:
            For `method='gaussian'` on forced-choice data, this also sets
            `gaussian_sensitivity`, `gaussian_specificity`, `gaussian_ppv`, and
            `gaussian_auc` from the fitted Gaussian model. For `method='gaussian'`
            on either kind of data, it also sets `tpr_smooth`, `fpr_smooth`, and
            `aucn` (the smoothed curve and its AUC). None of these attributes are
            read by `calculate`.
        """

        self.calculate(balanced_acc=balanced_acc)  # Calculate ROC parameters

        if method == "gaussian":
            if self.forced_choice is not None:
                positive_idx, negative_idx = _forced_choice_pairs(
                    self.forced_choice, self.binary_outcome
                )
                # Within-pair centering shifts both scores of a pair equally, so
                # the differences are the same on the raw scores.
                diff_scores = (
                    self.input_values[positive_idx] - self.input_values[negative_idx]
                )
                mn_diff = np.mean(diff_scores)
                d = mn_diff / np.std(diff_scores)
                pooled_sd = np.std(diff_scores) / np.sqrt(2)
                d_a_model = mn_diff / pooled_sd

                expected_acc = 1 - norm.cdf(0, d, 1)
                self.gaussian_sensitivity = expected_acc
                self.gaussian_specificity = expected_acc
                self.gaussian_ppv = self.gaussian_sensitivity / (
                    self.gaussian_sensitivity + 1 - self.gaussian_specificity
                )
                self.gaussian_auc = norm.cdf(d_a_model / np.sqrt(2))

                x = np.arange(-3, 3, 0.1)
                self.tpr_smooth = 1 - norm.cdf(x, d, 1)
                self.fpr_smooth = 1 - norm.cdf(x, -d, 1)
            else:
                mn_true = np.mean(self.input_values[self.binary_outcome])
                mn_false = np.mean(self.input_values[~self.binary_outcome])
                var_true = np.var(self.input_values[self.binary_outcome])
                var_false = np.var(self.input_values[~self.binary_outcome])
                pooled_sd = np.sqrt(
                    (var_true * (self.n_true - 1) + var_false * (self.n_false - 1))
                    / (self.n_true + self.n_false - 2)
                )
                d = (mn_true - mn_false) / pooled_sd
                z_true = mn_true / pooled_sd
                z_false = mn_false / pooled_sd

                x = np.arange(z_false - 3, z_true + 3, 0.1)
                self.tpr_smooth = 1 - (norm.cdf(x, z_true, 1))
                self.fpr_smooth = 1 - (norm.cdf(x, z_false, 1))

            self.aucn = auc(self.fpr_smooth, self.tpr_smooth)
            fig = _plot_roc(self.fpr_smooth, self.tpr_smooth)

        elif method == "observed":
            fig = _plot_roc(self.fpr, self.tpr)
        else:
            raise ValueError("method must be 'gaussian' or 'observed'")
        return fig

    def summary(self):
        """Display a formatted summary of ROC analysis."""

        print("------------------------")
        print(".:ROC Analysis Summary:.")
        print("------------------------")
        print("{:20s}".format("Accuracy:") + f"{self.accuracy:.2f}")
        print("{:20s}".format("Accuracy SE:") + f"{self.accuracy_se:.2f}")
        print("{:20s}".format("Accuracy p-value:") + f"{self.accuracy_p.pvalue:.2f}")
        print("{:20s}".format("Sensitivity:") + f"{self.sensitivity:.2f}")
        print("{:20s}".format("Specificity:") + f"{self.specificity:.2f}")
        print("{:20s}".format("AUC:") + f"{self.auc:.2f}")
        print("{:20s}".format("PPV:") + f"{self.ppv:.2f}")
        print("------------------------")
