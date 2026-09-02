"""ROC (Receiver Operating Characteristic) analysis for single-interval classification.

These tools provide the ability to quickly run receiver operating characteristic
analyses on the output of machine-learning models applied to imaging data.
"""

__all__ = ["Roc"]

import numpy as np
from nltools.plotting import plot_roc
from scipy.stats import norm, binomtest
from sklearn.metrics import auc
from copy import deepcopy


class Roc:
    """Compute receiver operating characteristic curves for single-interval or forced-choice classification.

    The Roc class is based on Tor Wager's Matlab roc_plot.m function and
    allows a user to easily run different types of receiver operator
    characteristic curves.  For example, one might be interested in single
    interval or forced choice.

    Args:
        input_values (array-like): 1-D continuous decision values, one per observation.
        binary_outcome (array-like): Boolean class label per observation.
        method (str): Threshold-selection variant, one of `'optimal_overall'`,
            `'optimal_balanced'`, `'minimum_sdt_bias'`.
        forced_choice (array-like, optional): Subject id per observation for
            forced-choice classification (each subject contributes one positive and
            one negative observation).

    Attributes:
        input_values (np.ndarray): Decision values.
        binary_outcome (np.ndarray): Boolean labels.
        method (str): Threshold-selection variant.
        forced_choice (np.ndarray | None): Subject ids for forced-choice classification.
        criterion_values (np.ndarray): Thresholds at which `tpr`/`fpr` were evaluated;
            set by `calculate`.
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
    """

    def __init__(
        self,
        *,
        input_values=None,
        binary_outcome=None,
        method="optimal_overall",
        forced_choice=None,
    ):
        if len(input_values) != len(binary_outcome):
            raise ValueError(
                "Data Problem: input_value and binary_outcomeare different lengths."
            )

        binary_outcome = np.asarray(binary_outcome).astype(bool).flatten()
        if binary_outcome.all() or not binary_outcome.any():
            raise ValueError(
                "Data Problem: binary_outcome must contain both positive and "
                "negative cases (True and False)."
            )

        valid_methods = ["optimal_overall", "optimal_balanced", "minimum_sdt_bias"]
        if method not in valid_methods:
            raise ValueError(
                "method must be ['optimal_overall', "
                "'optimal_balanced','minimum_sdt_bias']"
            )

        self.input_values = np.array(input_values)
        self.method = deepcopy(method)
        self.forced_choice = deepcopy(forced_choice)
        self.binary_outcome = binary_outcome

    def calculate(
        self,
        *,
        input_values=None,
        binary_outcome=None,
        criterion_values=None,
        method="optimal_overall",
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
                `fpr` and `tpr`. Defaults to a dense grid over the range of
                `input_values`.
            method (str): Threshold-selection variant, one of `'optimal_overall'`,
                `'optimal_balanced'`, `'minimum_sdt_bias'`.
            forced_choice (array-like, optional): Subject id per observation for
                forced-choice classification.
            balanced_acc (bool): Report balanced accuracy (mean of sensitivity and
                specificity) instead of overall accuracy. Only affects the accuracy
                estimate, not the p-value or the threshold used for
                sensitivity/specificity.
            tail (int | str): `2`/`'two'` for two-tailed (default); `1`/`'one'` for
                one-tailed (accuracy > chance) in the binomial test for `accuracy_p`.
        """
        from nltools.algorithms.inference.validation import validate_tail_parameter

        binom_alternative = (
            "two-sided" if validate_tail_parameter(tail) == "two" else "greater"
        )

        if input_values is not None:
            self.input_values = np.array(input_values)

        if binary_outcome is not None:
            self.binary_outcome = np.asarray(binary_outcome).astype(bool).flatten()

        # Create Criterion Values
        if criterion_values is not None:
            self.criterion_values = deepcopy(criterion_values)
        else:
            self.criterion_values = np.linspace(
                np.min(self.input_values.squeeze()),
                np.max(self.input_values.squeeze()),
                num=50 * len(self.binary_outcome),
            )

        if forced_choice is not None:
            self.forced_choice = deepcopy(forced_choice)

        if self.forced_choice is not None:
            sub_idx = np.unique(self.forced_choice)
            if len(sub_idx) != len(self.binary_outcome) / 2:
                raise ValueError(
                    "Make sure that subject ids are correct for 'forced_choice'."
                )
            if len(
                set(sub_idx).union(
                    set(np.array(self.forced_choice)[self.binary_outcome])
                )
            ) != len(sub_idx):
                raise ValueError("Issue with forced_choice subject labels.")
            if len(
                set(sub_idx).union(
                    set(np.array(self.forced_choice)[~self.binary_outcome])
                )
            ) != len(sub_idx):
                raise ValueError("Issue with forced_choice subject labels.")
            for sub in sub_idx:
                sub_mn = (
                    self.input_values[
                        (self.forced_choice == sub) & (self.binary_outcome)
                    ]
                    + self.input_values[
                        (self.forced_choice == sub) & (~self.binary_outcome)
                    ]
                )[0] / 2
                self.input_values[
                    (self.forced_choice == sub) & (self.binary_outcome)
                ] = (
                    self.input_values[
                        (self.forced_choice == sub) & (self.binary_outcome)
                    ][0]
                    - sub_mn
                )
                self.input_values[
                    (self.forced_choice == sub) & (~self.binary_outcome)
                ] = (
                    self.input_values[
                        (self.forced_choice == sub) & (~self.binary_outcome)
                    ][0]
                    - sub_mn
                )
            self.class_thr = 0

        # Calculate true positive and false positive rate
        self.tpr = np.zeros(self.criterion_values.shape)
        self.fpr = np.zeros(self.criterion_values.shape)
        for i, x in enumerate(self.criterion_values):
            wh = self.input_values >= x
            self.tpr[i] = np.sum(wh[self.binary_outcome]) / np.sum(self.binary_outcome)
            self.fpr[i] = np.sum(wh[~self.binary_outcome]) / np.sum(
                ~self.binary_outcome
            )
        self.n_true = np.sum(self.binary_outcome)
        self.n_false = np.sum(~self.binary_outcome)
        self.auc = auc(self.fpr, self.tpr)

        # Get criterion threshold
        if self.forced_choice is None:
            self.method = method
            if method == "optimal_balanced":
                mn = (self.tpr + self.fpr) / 2
                self.class_thr = self.criterion_values[np.argmax(mn)]
            elif method == "optimal_overall":
                n_corr_t = self.tpr * self.n_true
                n_corr_f = (1 - self.fpr) * self.n_false
                sm = n_corr_t + n_corr_f
                self.class_thr = self.criterion_values[np.argmax(sm)]
            elif method == "minimum_sdt_bias":
                # Calculate  MacMillan and Creelman 2005 Response Bias (c_bias)
                c_bias = (
                    norm.ppf(np.maximum(0.0001, np.minimum(0.9999, self.tpr)))
                    + norm.ppf(np.maximum(0.0001, np.minimum(0.9999, self.fpr)))
                ) / float(2)
                self.class_thr = self.criterion_values[np.argmin(abs(c_bias))]

        # Calculate output
        self.false_positive = (self.input_values >= self.class_thr) & (
            ~self.binary_outcome
        )
        self.false_negative = (self.input_values < self.class_thr) & (
            self.binary_outcome
        )
        self.misclass = (self.false_negative) | (self.false_positive)
        self.true_positive = (self.binary_outcome) & (~self.misclass)
        self.true_negative = (~self.binary_outcome) & (~self.misclass)
        self.sensitivity = (
            np.sum(self.input_values[self.binary_outcome] >= self.class_thr)
            / self.n_true
        )
        self.specificity = (
            1
            - np.sum(self.input_values[~self.binary_outcome] >= self.class_thr)
            / self.n_false
        )
        self.ppv = np.sum(self.true_positive) / (
            np.sum(self.true_positive) + np.sum(self.false_positive)
        )
        if self.forced_choice is not None:
            self.true_positive = self.true_positive[self.binary_outcome]
            self.true_negative = self.true_negative[~self.binary_outcome]
            self.false_negative = self.false_negative[self.binary_outcome]
            self.false_positive = self.false_positive[~self.binary_outcome]
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
        to the decision values or the observed empirical curve.

        Args:
            method (str): Type of plot, `'gaussian'` or `'observed'`.
            balanced_acc (bool): Passed to `calculate`; report balanced accuracy.

        Returns:
            matplotlib.figure.Figure: The ROC figure.
        """

        self.calculate(balanced_acc=balanced_acc)  # Calculate ROC parameters

        if method == "gaussian":
            if self.forced_choice is not None:
                sub_idx = np.unique(self.forced_choice)
                diff_scores = []
                for sub in sub_idx:
                    diff_scores.append(
                        self.input_values[
                            (self.forced_choice == sub) & (self.binary_outcome)
                        ][0]
                        - self.input_values[
                            (self.forced_choice == sub) & (~self.binary_outcome)
                        ][0]
                    )
                diff_scores = np.array(diff_scores)
                mn_diff = np.mean(diff_scores)
                d = mn_diff / np.std(diff_scores)
                pooled_sd = np.std(diff_scores) / np.sqrt(2)
                d_a_model = mn_diff / pooled_sd

                expected_acc = 1 - norm.cdf(0, d, 1)
                self.sensitivity = expected_acc
                self.specificity = expected_acc
                self.ppv = self.sensitivity / (self.sensitivity + 1 - self.specificity)
                self.auc = norm.cdf(d_a_model / np.sqrt(2))

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
            fig = plot_roc(self.fpr_smooth, self.tpr_smooth)

        elif method == "observed":
            fig = plot_roc(self.fpr, self.tpr)
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
