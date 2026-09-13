# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "nltools>=0.6.0",
# ]
# ///
# Multivariate prediction — marimo notebook. Source of truth for the docs page; rendered to the docs page by scripts/marimo_to_zensical.py.

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
    # Multivariate Prediction

    A univariate test asks whether each voxel tracks a variable on its own.
    A multivariate model asks what the whole pattern predicts, and answers it
    with a number you can check: how well the model does on data it was not
    fitted on.

    `BrainData.predict` runs that whole loop — folds, fits, scores, and a final
    fit on everything — and hands back one frozen `Predict` record. This tutorial
    predicts pain intensity from 84 images of 28 subjects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the data

    Targets and grouping variables live in `.Y`, and `predict` takes them by
    column name: `y="PainLevel"` is what to predict, `groups="SubjectID"` is what
    the splitter keeps together.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from joblib import Memory

    from nltools.datasets import fetch_pain

    memory = Memory(".tutorial-cache", verbose=0)

    data = memory.cache(fetch_pain)()
    data.Y = data.X.select("PainLevel", "SubjectID")
    data.Y.head()
    return data, memory, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One cross-validated prediction

    The `"ridge"` shortcut standardizes every voxel inside each fold and then
    fits ridge regression. `GroupKFold` with `groups="SubjectID"` puts all three
    of a subject's images in the same fold, so the model is always scored on
    people it has never seen.

    Ridge's penalty has to match the size of the problem, so the `"ridge"`
    shortcut picks it by an inner cross-validation of each training fold rather
    than leaving it at scikit-learn's default of 1.
    """)
    return


@app.cell
def _(data, memory):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    subject_folds = GroupKFold(n_splits=5)

    @memory.cache
    def cross_validate(estimator, cv, groups="SubjectID"):
        """Cross-validate one estimator against pain level, across the brain.

        A thin wrapper on `predict` so the docs build reuses fits instead of
        refitting every estimator on every run. joblib keys the cache on this
        function's source and its arguments, not on `data`, which the body
        closes over — so change how the data is prepared and clear the cache, or
        the numbers below are the old ones.
        """
        return data.predict(y="PainLevel", estimator=estimator, cv=cv, groups=groups)

    ridge = cross_validate("ridge", subject_folds)
    ridge.available()
    return Ridge, StandardScaler, cross_validate, make_pipeline, ridge, subject_folds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Those are the stored fields a whole-brain run fills. `scores` holds one
    score per fold — R² here, because that is what a scikit-learn regressor's own
    `score` method reports. `mean_score` and `std_score` are not in that list:
    they are computed from `scores` on demand. `weight_map` is not an average of
    the folds either — it comes from one more fit on all 84 images, which is the
    model you would publish.
    """)
    return


@app.cell
def _(ridge):
    print(f"fold R²: {ridge.scores.round(3)}")
    print(f"mean R²: {ridge.mean_score:.3f} ± {ridge.std_score:.3f}")
    ridge.weight_map.plot(title="Ridge weights: predicting pain intensity")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `predictions` holds the out-of-fold prediction for every image, in the
    original row order, and `cv_folds` says which fold produced each one. Plot
    them against the truth and the model's real behaviour shows up: it orders the
    three intensities well, while compressing their range.
    """)
    return


@app.cell
def _(data, np, plt, ridge):
    observed = data.Y["PainLevel"].to_numpy()
    jitter = np.random.default_rng(0).normal(0, 0.04, observed.size)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(observed + jitter, ridge.predictions, alpha=0.6)
    axis.set_xticks([1, 2, 3], ["low", "medium", "high"])
    axis.set_xlabel("observed pain intensity")
    axis.set_ylabel("cross-validated prediction")
    axis.set_title(
        f"r = {np.corrcoef(ridge.predictions, observed)[0, 1]:.2f} across held-out subjects"
    )
    figure.tight_layout()
    figure
    return (observed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Other estimators

    `estimator=` takes seven shortcut names — `'ridge'`, `'lasso'`,
    `'linear_svr'` for regression, and `'linear_svc'`,
    `'logistic_regression'`, `'linear_discriminant_analysis'`,
    `'ridge_classifier'` for classification — each of which standardizes voxels
    inside every fold and then fits that estimator with scikit-learn's own
    defaults. Those defaults assume far fewer features than a brain has, so
    whole-brain work usually means passing a pipeline with the penalty set
    explicitly, as above.

    Any scikit-learn estimator or `Pipeline` is accepted and used exactly as
    supplied. Preprocessing steps may be scalers, PCA, or feature selectors; the
    final step must expose `coef_` so the weights can be projected back onto the
    voxels.

    Principal components regression is the classic case: reduce 240,000 voxels to
    a handful of components, then regress on those. Swapping the final step for a
    lasso gives LASSO-PCR.
    """)
    return


@app.cell
def _(Ridge, StandardScaler, cross_validate, make_pipeline, subject_folds):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.svm import SVR

    estimators = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1e5)),
        "support vector": make_pipeline(StandardScaler(), SVR(kernel="linear")),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
        "PCR": make_pipeline(
            StandardScaler(), PCA(n_components=10), LinearRegression()
        ),
        "LASSO-PCR": make_pipeline(
            StandardScaler(), PCA(n_components=10), Lasso(alpha=0.1)
        ),
    }
    fits = {
        name: cross_validate(estimator, subject_folds)
        for name, estimator in estimators.items()
    }
    return (fits,)


@app.cell
def _(fits, np, observed):
    print(f"{'estimator':16s} {'mean R²':>8s} {'r':>6s}")
    for estimator_name, fit in fits.items():
        correlation = np.corrcoef(fit.predictions, observed)[0, 1]
        print(f"{estimator_name:16s} {fit.mean_score:8.3f} {correlation:6.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    R² and correlation disagree, and the disagreement is informative. R² punishes
    a prediction that is systematically offset or compressed even when it orders
    the images correctly; correlation only asks about the ordering. Across
    held-out subjects, models that rank pain well often miss its absolute level,
    so report both.

    PCR and LASSO-PCR print the same numbers, and that is not a mistake. The
    `alpha=0.1` penalty is negligible against the scale of ten PCA scores, so the
    lasso shrinks essentially nothing and the two fits coincide; a penalty large
    enough to zero components would separate them.

    ## Cross-validation schemes

    `cv` follows scikit-learn's grammar. `cv=None` is a deterministic five-fold
    `KFold` — `StratifiedKFold` when the target is discrete — an integer is that
    many folds, and any scikit-learn splitter is used exactly as supplied.
    `predict` never shuffles on your behalf: when rows are ordered by condition,
    unshuffled contiguous folds are degenerate, and the fix is a splitter you
    construct with `shuffle=True` and a `random_state`.

    Four schemes on the same estimator:

    - **five-fold, no groups** — the default. Contiguous folds of 17, 17, 17, 17
      and 16 images keep most subjects together by accident, but three of the 28
      straddle a fold boundary, which is part of why this score differs from the
      grouped run. Nothing guarantees even that much.
    - **grouped five-fold** — each subject's three images stay together, so no
      subject is ever in both the training and the test set.
    - **stratified on a continuous target** — `KFoldStratified` orders rows by
      `y` and deals them round-robin, so every fold spans the full range of pain.
      Shuffled, with a seed, to break ties reproducibly. It balances the target
      and nothing else: with `groups=None` a held-out image's own subject is in
      the training set, so read its score as optimistic.
    - **leave one subject out** — `LeaveOneGroupOut` with the subject ids, so
      k equals the number of subjects.
    """)
    return


@app.cell
def _(Ridge, StandardScaler, cross_validate, make_pipeline, subject_folds):
    from sklearn.model_selection import LeaveOneGroupOut

    from nltools.cross_validation import KFoldStratified

    ridge_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1e5))
    schemes = {
        "five-fold": (None, None),
        "grouped five-fold": (subject_folds, "SubjectID"),
        "stratified on y": (
            KFoldStratified(n_splits=5, shuffle=True, random_state=0),
            None,
        ),
        "leave one subject out": (LeaveOneGroupOut(), "SubjectID"),
    }

    print(f"{'scheme':24s} {'folds':>5s} {'mean R²':>8s} {'sd':>6s}")
    for scheme_name, (splitter, scheme_groups) in schemes.items():
        scheme_fit = cross_validate(ridge_pipeline, splitter, scheme_groups)
        print(
            f"{scheme_name:24s} {len(scheme_fit.scores):5d} "
            f"{scheme_fit.mean_score:8.3f} {scheme_fit.std_score:6.2f}"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The stratified scheme posts the highest mean R² of the four, which is what
    subject leakage looks like: predicting a person's medium-pain image is easy
    once the model has seen their low and high ones. Only the two grouped rows
    answer the question the rest of this tutorial asks.

    Leaving one subject out gives 28 folds of three images each, so each fold's
    R² is estimated from almost nothing and the spread across folds is enormous.
    More folds buys more training data, not a more stable score.

    ### Choosing the penalty inside the loop

    `RidgeCV` picks `alpha` by its own inner cross-validation on each training
    split, which keeps the choice out of the test fold — the honest alternative
    to reading a grid of hand-set penalties off the outer score.
    """)
    return


@app.cell
def _(StandardScaler, cross_validate, make_pipeline, np, subject_folds):
    from sklearn.linear_model import RidgeCV

    ridgecv = cross_validate(
        make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(3, 7, 9))),
        subject_folds,
    )
    print(f"alpha chosen on all the data: {ridgecv.estimator[-1].alpha_:.0e}")
    print(f"mean R²: {ridgecv.mean_score:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It settles on a much weaker penalty than the one set by hand, and scores
    worse for it. Nothing is broken: `RidgeCV` splits each training set without
    knowing about subjects, so it judges a candidate penalty by how well it
    predicts other images from the *same* people — an easier problem than the one
    the outer folds pose. Nested selection removes a bias in the score; it does
    not choose a penalty for a generalization you never described to it.

    ## Recap

    | Step | Call |
    |---|---|
    | Attach targets and groups | `data.Y = data.X.select("PainLevel", "SubjectID")` |
    | Cross-validated prediction | `data.predict(y=, estimator=, cv=, groups=)` |
    | Fields the run filled | `result.available()` |
    | Honest score | `result.scores`, `result.mean_score`, `result.std_score` |
    | Publishable map | `result.weight_map` (fitted on all the data) |
    | Out-of-fold predictions | `result.predictions`, `result.cv_folds` |
    | Fitted estimator, for new data | `result.estimator` |

    **Next steps**

    - [Multivariate Classification](03_multivariate_classification.md) — the same
      machinery with discrete labels, plus ROC analysis.
    - [Multivariate Pattern Analysis](../workflows/03_mvpa.md) — the same call at
      ROI and searchlight scales.
    """)
    return


if __name__ == "__main__":
    app.run()
