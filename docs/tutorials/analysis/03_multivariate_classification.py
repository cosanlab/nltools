# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Multivariate classification — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multivariate Classification

    Classification is prediction with a discrete target: train a model to tell
    two conditions apart from the whole activity pattern, and score it on people
    it has not seen. The call is the same `BrainData.predict`; what changes is the
    target, the estimators, and how you evaluate the result.

    Here the two conditions are the high- and low-intensity pain images of 28
    subjects, and the evaluation runs past accuracy to a full receiver operating
    characteristic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare the two classes

    Take each subject's high- and low-pain image, stack them, and build the
    target: 1 for high, 0 for low. `groups` carries the subject id for each row,
    so the splitter can hold a subject's pair out together.
    """)
    return


@app.cell
def _():
    import numpy as np
    from joblib import Memory

    from nltools.data import Roc
    from nltools.datasets import fetch_pain
    from nltools.utils import concatenate

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    high = data[data.X["PainLevel"] == 3]
    low = data[data.X["PainLevel"] == 1]

    pairs = concatenate([high, low])
    labels = np.r_[np.ones(len(high)), np.zeros(len(low))].astype(int)
    subject_ids = np.r_[high.X["SubjectID"].to_numpy(), low.X["SubjectID"].to_numpy()]

    print(f"images: {pairs.shape}   high: {labels.sum()}   low: {(1 - labels).sum()}")
    return Roc, labels, memory, np, pairs, subject_ids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classify, cross-validated

    `'linear_svc'` is one of the seven estimator shortcuts: a `StandardScaler`
    over voxels inside each fold, then a linear support vector classifier.
    `GroupKFold` with the subject ids keeps each subject's pair in the same fold.

    The score is accuracy, because that is what a scikit-learn classifier's own
    `score` method reports. `weight_map` is signed for the second class against
    the first — positive voxels push the model toward *high*.
    """)
    return


@app.cell
def _(labels, memory, pairs, subject_ids):
    from sklearn.model_selection import GroupKFold

    subject_folds = GroupKFold(n_splits=5)

    @memory.cache
    def decode(estimator):
        """Cross-validate one classifier on high versus low pain.

        A thin wrapper on `predict` so the docs build reuses fits instead of
        refitting every classifier on every run. joblib keys the cache on this
        function's source and its argument, not on the data, folds and labels the
        body closes over — so change how those are built and clear the cache, or
        the scores below are the old ones.
        """
        return pairs.predict(
            y=labels, estimator=estimator, cv=subject_folds, groups=subject_ids
        )

    svm = decode("linear_svc")
    print(f"classes: {svm.classes}")
    print(f"accuracy: {svm.mean_score:.3f} ± {svm.std_score:.3f}   (chance 0.5)")
    svm.weight_map.plot(title="SVM weights: + favors high pain, − favors low")
    return decode, subject_folds, svm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Other classifiers

    Logistic regression and a ridge classifier are two more shortcuts. Ridge, as
    in the regression tutorial, needs a penalty scaled to 240,000 voxels rather
    than scikit-learn's default of 1, so it goes in as an explicit pipeline.
    """)
    return


@app.cell
def _(decode):
    from sklearn.linear_model import RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    classifiers = {
        "support vector": "linear_svc",
        "logistic regression": "logistic_regression",
        "ridge classifier": make_pipeline(StandardScaler(), RidgeClassifier(alpha=1e5)),
    }
    for classifier_name, classifier in classifiers.items():
        fit = decode(classifier)
        print(f"{classifier_name:20s} {fit.mean_score:.3f} ± {fit.std_score:.3f}")
    return SVC, StandardScaler, make_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Continuous decision values

    `Predict.predictions` holds the out-of-fold *label* for every image — what
    the classifier decided. Calibration and ROC analysis need the continuous
    quantity behind that decision: the distance from the separating hyperplane,
    or a probability.

    `result.estimator` is the fitted pipeline, so scikit-learn's
    `cross_val_predict` will produce either one, over the same folds.
    """)
    return


@app.cell
def _(labels, memory, pairs, subject_folds, subject_ids, svm):
    from sklearn.model_selection import cross_val_predict

    @memory.cache
    def out_of_fold(estimator, method):
        """Out-of-fold continuous output, over the folds used above.

        Cached on the same terms as `decode`: the data, folds and labels it
        closes over are not part of the key.
        """
        return cross_val_predict(
            estimator,
            pairs.data,
            labels,
            cv=subject_folds,
            groups=subject_ids,
            method=method,
        )

    decisions = out_of_fold(svm.estimator, "decision_function")
    print(f"distance from the hyperplane, first six images: {decisions[:6].round(2)}")
    return decisions, out_of_fold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Probabilities by Platt scaling

    A support vector classifier has no probability of its own. `SVC` with
    `probability=True` fits a logistic curve to the decision values by internal
    cross-validation — Platt scaling — and gains `predict_proba`.
    """)
    return


@app.cell
def _(SVC, StandardScaler, labels, make_pipeline, np, out_of_fold):
    platt = make_pipeline(
        StandardScaler(), SVC(kernel="linear", probability=True, random_state=0)
    )
    probabilities = out_of_fold(platt, "predict_proba")[:, 1]

    print(f"mean P(high) for high-pain images: {probabilities[labels == 1].mean():.2f}")
    print(f"mean P(high) for low-pain images:  {probabilities[labels == 0].mean():.2f}")
    print(f"agreement with the labels: {np.mean((probabilities > 0.5) == labels):.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ROC analysis

    Accuracy hides the trade-off a classifier is making: a model can be accurate
    because it is sensitive, or because it is specific, and the two fail
    differently. A receiver operating characteristic sweeps the decision
    threshold and plots sensitivity against one minus specificity at every point.

    `Roc` takes the continuous values and the boolean outcome. `plot` runs the
    calculation and draws the curve; `summary` reports the numbers at the chosen
    threshold. `calculate` alone does the work without drawing anything.
    """)
    return


@app.cell
def _(Roc, decisions, labels):
    roc = Roc(input_values=decisions, binary_outcome=labels.astype(bool))
    roc.plot()
    roc.summary()
    print(
        f"single interval: accuracy {roc.accuracy:.2f}   "
        f"sensitivity {roc.sensitivity:.2f}   "
        f"specificity {roc.specificity:.2f}   AUC {roc.auc:.2f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `method` chooses the rule that sets the threshold: `'optimal_overall'`
    weights every observation equally, `'optimal_balanced'` weights the two
    classes equally, and `'minimum_sdt_bias'` picks the point of least response
    bias. `calculate` uses whatever the constructor was given; a `method=` passed
    to `calculate` overrides it for that one call and does not stick.

    With equal numbers of high- and low-pain images the first two rules are the
    same question, so they land on the same threshold.
    """)
    return


@app.cell
def _(Roc, decisions, labels):
    for rule in ["optimal_overall", "optimal_balanced", "minimum_sdt_bias"]:
        rule_roc = Roc(
            input_values=decisions,
            binary_outcome=labels.astype(bool),
            method=rule,
        )
        rule_roc.calculate()
        print(
            f"{rule:18s} threshold {rule_roc.class_thr:6.3f}   "
            f"sensitivity {rule_roc.sensitivity:.2f}   "
            f"specificity {rule_roc.specificity:.2f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Forced choice

    Everything above is *single-interval* classification: each image is judged on
    its own against one threshold, so a subject whose activity is globally high
    can be called high-pain twice. Often the question is comparative instead —
    given this person's two images, which one is the high-pain one? That is
    *forced-choice* classification, and it needs no threshold at all: the larger
    of the two decision values wins.

    Pass `forced_choice` a subject id per observation. Each subject must
    contribute exactly one positive and one negative observation, which is how
    these pairs were built.
    """)
    return


@app.cell
def _(Roc, decisions, labels, subject_ids):
    forced_choice = Roc(
        input_values=decisions,
        binary_outcome=labels.astype(bool),
        forced_choice=subject_ids,
    )
    forced_choice.plot()
    forced_choice.summary()
    print(
        f"forced choice:   accuracy {forced_choice.accuracy:.2f}   "
        f"sensitivity {forced_choice.sensitivity:.2f}   "
        f"specificity {forced_choice.specificity:.2f}   "
        f"AUC {forced_choice.auc:.2f}"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Within-person comparison is the easier question, and the accuracy shows it.
    The per-subject offsets that cost the single-interval classifier its mistakes
    cancel when both images come from the same person.

    ## Recap

    | Step | Call |
    |---|---|
    | Stack the two classes | `concatenate([high, low])` |
    | Cross-validated classification | `pairs.predict(y=labels, estimator=, cv=, groups=)` |
    | Accuracy and its spread | `result.mean_score`, `result.std_score` |
    | Signed pattern | `result.weight_map` |
    | Out-of-fold labels | `result.predictions` |
    | Out-of-fold decision values | `cross_val_predict(result.estimator, ..., method="decision_function")` |
    | Probabilities | `SVC(probability=True)`, then `method="predict_proba"` |
    | Single-interval ROC | `Roc(input_values=, binary_outcome=)` |
    | Forced choice | `Roc(..., forced_choice=subject_ids)` |

    **Next steps**

    - [Similarity and Distance](04_similarity.md) — compare patterns to each
      other instead of fitting a model.
    - [Multivariate Pattern Analysis](../workflows/03_mvpa.md) — the same call at
      ROI and searchlight scales.
    """)
    return


if __name__ == "__main__":
    app.run()
