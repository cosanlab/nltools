"""BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: `predict`. It resolves exactly one mode, validates every
argument for that mode, and returns either a new `BrainData` (fitted-model
prediction) or a frozen `Predict` record (MVPA). Nothing is attached to the
source object.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from nltools.algorithms.decoding import (
    back_project_weight_maps,
    validate_decoding_pipeline,
)
from nltools.data.results import Predict
from nltools.utils import maybe_tqdm

from .utils import _is_default


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


#: Decoding-only arguments and their documented defaults. A non-default value
#: for any of them on a fitted-model call is an invalid combination, not a
#: silently ignored keyword.
MVPA_ONLY_DEFAULTS = {
    "estimator": "linear_svc",
    "cv": None,
    "groups": None,
    "scoring": None,
    "spatial_scale": "whole_brain",
    "roi_mask": None,
    "radius": 10.0,
    "n_jobs": 1,
    "progress_bar": False,
}


def predict(
    bd,
    *,
    X=None,
    y=None,
    estimator: Any = "linear_svc",
    cv=None,
    groups=None,
    scoring=None,
    spatial_scale: str = "whole_brain",
    roi_mask=None,
    radius: float = 10.0,
    n_jobs: int = 1,
    progress_bar: bool = False,
):
    """Dispatch BrainData prediction to fitted-model prediction or MVPA decoding.

    Implements `BrainData.predict`. See that method's docstring for full
    parameter documentation.
    """
    if X is not None and y is not None:
        raise ValueError(
            "Cannot specify both X and y. Use X to predict from a fitted "
            "model or y to decode with MVPA."
        )

    decoding_arguments = {
        "estimator": estimator,
        "cv": cv,
        "groups": groups,
        "scoring": scoring,
        "spatial_scale": spatial_scale,
        "roi_mask": roi_mask,
        "radius": radius,
        "n_jobs": n_jobs,
        "progress_bar": progress_bar,
    }

    if X is not None:
        _reject_decoding_arguments(decoding_arguments)
        return predict_timeseries(bd, X=X)

    resolved_y = _resolve_stored_y(bd, y)
    if resolved_y is None:
        # No labels to decode: the only remaining mode is prediction from a
        # fitted model, which takes none of the decoding arguments.
        _reject_decoding_arguments(decoding_arguments)
        return predict_timeseries(bd, X=None)

    return predict_mvpa(
        bd,
        y=resolved_y,
        estimator=estimator,
        cv=cv,
        groups=_resolve_stored_groups(bd, groups),
        scoring=scoring,
        spatial_scale=spatial_scale,
        roi_mask=roi_mask,
        radius=radius,
        n_jobs=n_jobs,
        progress_bar=progress_bar,
    )


def _reject_decoding_arguments(supplied: dict) -> None:
    """Raise when a decoding-only argument is passed on a call that is not decoding."""
    offenders = sorted(
        name
        for name, value in supplied.items()
        if not _is_default(value, MVPA_ONLY_DEFAULTS[name])
    )
    if not offenders:
        return
    names = ", ".join(f"{name}=" for name in offenders)
    verb = "configures" if len(offenders) == 1 else "configure"
    subject = "this argument" if len(offenders) == 1 else "these arguments"
    raise ValueError(
        f"{names} only {verb} MVPA decoding, and this call is not decoding. "
        f"Pass y= — or attach labels to .Y — to decode, or drop {subject}."
    )


# ---------------------------------------------------------------------------
# Stored-Y resolution (labels travel with the data)
# ---------------------------------------------------------------------------


def _series_to_numpy(series):
    """Convert a polars Series to numpy, mapping string columns to ``'<U'``.

    Polars Utf8 columns come back from ``to_numpy()`` as object arrays;
    sklearn then propagates the object dtype into ``classes_`` and
    ``predict()`` outputs, which breaks HDF5 persistence and dtype checks.
    A real unicode dtype keeps label arrays first-class end to end.
    """
    import polars as pl

    arr = series.to_numpy()
    if arr.dtype == object and series.dtype == pl.String:
        return arr.astype(str)
    return arr


def _resolve_stored_y(bd, y):
    """Resolve ``y`` against the stored ``bd.Y`` frame.

    Rules:
      - array-like ``y`` passes through as an ndarray;
      - a string picks that column of ``bd.Y``;
      - ``None`` falls back to a single-column ``bd.Y`` (the idiomatic
        labels-travel-with-the-data path). A multi-column ``Y`` is ambiguous
        and asks for ``y='name'``; an empty ``Y`` returns ``None`` so the
        dispatcher can fall through to timeseries prediction. A *fitted*
        encoding model wins over stored labels, so an object carrying both
        predicts its training timeseries.
    """
    stored = bd.Y

    if isinstance(y, str):
        if stored is None or stored.is_empty():
            raise ValueError(
                f"y={y!r} names a column of .Y, but no Y frame is stored on "
                f"this BrainData. Set brain.Y or pass y as an array."
            )
        if y not in stored.columns:
            raise ValueError(
                f"y={y!r} is not a column of .Y (columns: {stored.columns})."
            )
        return _series_to_numpy(stored[y])

    if y is not None:
        return np.asarray(y)

    if stored is None or stored.is_empty():
        return None

    if getattr(getattr(bd, "model_", None), "is_fitted_", False):
        # Fitted-model prediction wins over attached labels on a no-argument
        # call; returning None lets the dispatcher fall through to it. An
        # unfitted `model_` is not a model to predict from, so decoding the
        # stored labels stays available.
        return None
    if stored.shape[1] != 1:
        raise ValueError(
            f".Y has {stored.shape[1]} columns ({stored.columns}); pass "
            f"y='name' to pick the label column."
        )
    return _series_to_numpy(stored[stored.columns[0]])


def _resolve_stored_groups(bd, groups):
    """Resolve a string ``groups`` spec to that column of ``bd.Y``.

    The ``Y`` frame is the row-aligned metadata carrier on ``BrainData``, so
    within-subject grouping variables (run, session, block) live there
    alongside the labels. Arrays and ``None`` pass through unchanged.
    """
    if not isinstance(groups, str):
        return groups
    stored = bd.Y
    if stored is None or stored.is_empty():
        raise ValueError(
            f"groups={groups!r} names a column of .Y, but no Y frame is "
            f"stored on this BrainData. Set brain.Y or pass groups as an array."
        )
    if groups not in stored.columns:
        raise ValueError(
            f"groups={groups!r} is not a column of .Y (columns: {stored.columns})."
        )
    return _series_to_numpy(stored[groups])


# ---------------------------------------------------------------------------
# Timeseries prediction (encoding model — uses fitted ridge / glm)
# ---------------------------------------------------------------------------


def predict_timeseries(bd, *, X=None):
    """Predict voxel timeseries from a fitted encoding model.

    Returns a fresh ``BrainData`` whose ``.data`` is the predicted timeseries.
    Encoding model prediction yields a brain image — the natural container is
    ``BrainData``, so it composes directly with downstream methods (`.plot()`,
    `.standardize()`, etc.). MVPA decoding (``y=`` mode) returns ``Predict``.

    With no ``X``, the fitted model returns an independent copy of the stored
    training predictions and keeps their row metadata: ``glm_predicted`` for a
    GLM, ``ridge_fitted_values`` for a Ridge. Neither retains the training
    features, so a no-argument call never refits or re-multiplies. With an
    explicit ``X``, structural validation and alignment belong to the
    estimator's own ``predict`` — named design columns for `Glm`, named feature
    spaces for a banded `Ridge` — and the result clears the source row metadata.
    """
    from nltools.models import Glm

    from .utils import _result_from_array

    if not hasattr(bd, "model_"):
        raise ValueError(
            "Must call fit() before predict() for timeseries prediction. "
            "Example: brain_data.fit(model='ridge', X=features)"
        )
    if not bd.model_.is_fitted_:
        raise ValueError("Model is not fitted")

    if X is not None:
        return _result_from_array(bd, bd.model_.predict(X), rows="clear")

    stored = bd.glm_predicted if isinstance(bd.model_, Glm) else bd.ridge_fitted_values
    return _result_from_array(bd, np.array(stored.data, copy=True), rows="preserve")


# ---------------------------------------------------------------------------
# MVPA decoding
# ---------------------------------------------------------------------------


VALID_SPATIAL_SCALES = {"whole_brain", "searchlight", "roi"}


def predict_mvpa(
    bd,
    *,
    y,
    estimator: Any,
    cv,
    groups,
    scoring,
    spatial_scale: str,
    roi_mask,
    radius: float,
    n_jobs: int,
    progress_bar: bool,
) -> Predict:
    """Run cross-validated decoding on a `BrainData` and return a `Predict`.

    Every argument is validated, and the cross-validation folds are
    materialized and checked, before a single model is fitted.
    """
    from sklearn.base import is_classifier

    _validate_spatial_scale(spatial_scale, roi_mask=roi_mask, radius=radius)
    y = _validate_target(y, n_rows=bd.shape[0])
    groups = _validate_groups(groups, n_rows=bd.shape[0])
    validate_scoring(scoring)

    pipe = build_pipeline(estimator, y=y)
    validate_decoding_pipeline(pipe)
    classifier = is_classifier(pipe)
    splits = resolve_splits(cv, X=bd.data, y=y, groups=groups, classifier=classifier)
    classes = np.unique(y) if classifier else None

    X_data = bd.data  # (n_samples, n_voxels)

    if spatial_scale == "whole_brain":
        return _run_whole_brain(
            bd,
            X_data,
            y,
            pipe,
            splits=splits,
            scoring=scoring,
            classes=classes,
            n_jobs=n_jobs,
        )
    if spatial_scale == "searchlight":
        return _run_searchlight(
            bd,
            X_data,
            y,
            pipe,
            splits=splits,
            scoring=scoring,
            classes=classes,
            radius=radius,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
    return _run_roi(
        bd,
        X_data,
        y,
        pipe,
        splits=splits,
        scoring=scoring,
        classes=classes,
        roi_mask=roi_mask,
        n_jobs=n_jobs,
        progress_bar=progress_bar,
    )


# ---------------------------------------------------------------------------
# Argument validation — everything here runs before the first fit
# ---------------------------------------------------------------------------


def _validate_spatial_scale(spatial_scale: str, *, roi_mask, radius: float) -> None:
    """Check the spatial scale and the companion arguments it owns."""
    if spatial_scale not in VALID_SPATIAL_SCALES:
        raise ValueError(
            f"Invalid spatial_scale: {spatial_scale!r}. "
            f"Must be one of {sorted(VALID_SPATIAL_SCALES)}"
        )
    if spatial_scale == "roi" and roi_mask is None:
        raise ValueError("roi_mask is required for spatial_scale='roi'")
    if spatial_scale != "roi" and roi_mask is not None:
        raise ValueError(
            f"roi_mask only applies to spatial_scale='roi', not {spatial_scale!r}."
        )
    if spatial_scale != "searchlight" and not _is_default(radius, 10.0):
        raise ValueError(
            f"radius only applies to spatial_scale='searchlight', not "
            f"{spatial_scale!r}."
        )


def _validate_target(y, *, n_rows: int) -> np.ndarray:
    """Return `y` as a one-dimensional array with one value per row."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(
            f"y must be one-dimensional with one value per row; got shape "
            f"{y.shape}. Multioutput and multilabel targets are not accepted."
        )
    if y.shape[0] != n_rows:
        raise ValueError(
            f"y must have one value per row: got {y.shape[0]} values for {n_rows} rows."
        )
    return y


def _validate_groups(groups, *, n_rows: int):
    """Return `groups` as a one-dimensional array with one value per row."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.shape[0] != n_rows:
        raise ValueError(
            f"groups must have one value per row: got shape {groups.shape} "
            f"for {n_rows} rows."
        )
    return groups


def validate_scoring(scoring) -> None:
    """Reject the removed `'auto'` value and multimetric scoring mappings."""
    from collections.abc import Mapping

    if isinstance(scoring, Mapping) or (
        isinstance(scoring, (list, tuple, set)) and not isinstance(scoring, str)
    ):
        raise ValueError(
            "Multimetric scoring is not accepted because Predict.scores holds "
            "one value per cross-validation fold. Pass a single scoring name "
            "or callable, or None to use the estimator's own score method."
        )
    if scoring == "auto":
        raise ValueError(
            "scoring='auto' was removed. Pass None (the default) to use the "
            "estimator's own score method — accuracy for a classifier, R2 for "
            "a regressor — or any scikit-learn scoring name or callable."
        )


# ---------------------------------------------------------------------------
# Estimator resolution and pipeline construction
# ---------------------------------------------------------------------------

#: Built-in shortcuts. Every one is a linear estimator, so its coefficients
#: project back to the voxel axis.
ESTIMATOR_SHORTCUTS = (
    "linear_svc",
    "logistic_regression",
    "linear_discriminant_analysis",
    "ridge_classifier",
    "ridge",
    "lasso",
    "linear_svr",
)

#: Abbreviations that used to name a shortcut. They are ambiguous — 'svm' and
#: 'svr' say nothing about the kernel, 'ridge' already means the regressor —
#: so they are rejected with the canonical spelling.
REJECTED_ABBREVIATIONS = {
    "svm": "linear_svc",
    "logistic": "logistic_regression",
    "lda": "linear_discriminant_analysis",
    "svr": "linear_svr",
}


def resolve_estimator(estimator: Any):
    """Resolve a shortcut name to an estimator, or pass an sklearn object through."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import Lasso, LogisticRegression, Ridge, RidgeClassifier
    from sklearn.svm import LinearSVC, LinearSVR

    builders = {
        "linear_svc": lambda: LinearSVC(dual="auto", max_iter=10000),
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "linear_discriminant_analysis": lambda: LinearDiscriminantAnalysis(),
        "ridge_classifier": lambda: RidgeClassifier(),
        "ridge": lambda: Ridge(),
        "lasso": lambda: Lasso(),
        "linear_svr": lambda: LinearSVR(),
    }

    if isinstance(estimator, str):
        if estimator in REJECTED_ABBREVIATIONS:
            canonical = REJECTED_ABBREVIATIONS[estimator]
            raise ValueError(
                f"estimator={estimator!r} is ambiguous and is not accepted; "
                f"use {canonical!r}."
            )
        if estimator not in builders:
            raise ValueError(
                f"Unknown estimator shortcut: {estimator!r}. Valid shortcuts: "
                f"{list(ESTIMATOR_SHORTCUTS)}, or pass any sklearn estimator."
            )
        return builders[estimator]()

    if not (hasattr(estimator, "fit") and hasattr(estimator, "predict")):
        raise TypeError(
            f"estimator must be a shortcut name or an object with fit/predict; "
            f"got {type(estimator).__name__}"
        )
    return estimator


def build_pipeline(estimator: Any, *, y: np.ndarray) -> Any:
    """Build the per-fold pipeline for `estimator`.

    A built-in shortcut selects a predefined pipeline: `StandardScaler` inside
    each fold, then the linear estimator the shortcut names. A classification
    shortcut on a multiclass target is wrapped in `OneVsRestClassifier`, which
    gives one signed coefficient row per class instead of whatever multiclass
    strategy the estimator happens to default to.

    A caller-supplied estimator or `Pipeline` is used exactly as given — MVPA
    adds, removes, and reconfigures nothing, and never overrides its multiclass
    strategy. Callers who want one-vs-rest supply a `OneVsRestClassifier`.

    Args:
        estimator: A shortcut name or an sklearn estimator/`Pipeline`.
        y: The validated target vector, used only to decide whether a
            classification shortcut faces a multiclass problem.

    Returns:
        The estimator to clone and fit in every fold.
    """
    from sklearn.base import is_classifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    resolved = resolve_estimator(estimator)
    if not isinstance(estimator, str):
        return resolved
    if is_classifier(resolved) and len(np.unique(y)) > 2:
        resolved = OneVsRestClassifier(resolved)
    return make_pipeline(StandardScaler(), resolved)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def resolve_splits(cv, *, X, y, groups, classifier: bool) -> list:
    """Resolve `cv` into materialized train/test splits and check the partition.

    Materializing once means every runner — and every parallel worker — sees
    the same folds, and it lets the partition rule be checked before any model
    is fitted.
    """
    splitter = _resolve_splitter(
        cv, classifier=classifier, grouped=groups is not None, n_rows=len(y)
    )
    splits = [
        (_as_indices(train), _as_indices(test))
        for train, test in _iter_split(splitter, X, y, groups)
    ]
    _validate_partition(splits, n_rows=len(y))
    return splits


def _resolve_splitter(cv, *, classifier: bool, grouped: bool, n_rows: int):
    """Turn a `cv` spec into a scikit-learn splitter following sklearn's grammar.

    `None` and an int both mean that many *stratified*, unshuffled folds. What
    they stratify on, and whether they keep a group whole, depends on the
    model and on whether the caller supplied `groups`:

    | model      | `groups` | splitter                                       |
    | ---------- | -------- | ---------------------------------------------- |
    | classifier | no       | `StratifiedKFold(n)` on the class labels        |
    | classifier | yes      | `StratifiedGroupKFold(n)` on the class labels   |
    | regressor  | no       | `StratifiedKFold(n)` on quantile bins of `y`    |
    | regressor  | yes      | `StratifiedGroupKFold(n)` on quantile bins of `y` |

    A plain `(Stratified)KFold` accepts `groups` in `split()` but ignores it,
    so before the group-aware rows a subject could straddle the train/test
    boundary while the caller believed otherwise. There is no shuffle and no
    `random_state` on any of these paths: `cv=None` and an int stay
    reproducible across calls. A supplied splitter is used exactly as given,
    and the `'loo'`/`'logo'` string aliases raise.

    Args:
        cv: `None`, an int fold count, or a scikit-learn splitter.
        classifier: Whether the resolved model is a classifier.
        grouped: Whether the caller supplied `groups`.
        n_rows: The number of observations to be split.

    Returns:
        A scikit-learn-compatible splitter.

    Raises:
        ValueError: On a string `cv`, or on a regressor with too few rows to
            fill the quantile bins.
        TypeError: On anything that is not `None`, an int, or a splitter.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if isinstance(cv, str):
        raise ValueError(
            f"cv={cv!r} is not accepted: the 'loo' and 'logo' aliases were "
            f"removed from predict. Pass the splitter itself — "
            f"LeaveOneOut() or LeaveOneGroupOut() with groups= — an int fold "
            f"count, or None for a deterministic five-fold split."
        )
    if cv is None or (isinstance(cv, int) and not isinstance(cv, bool)):
        n_splits = 5 if cv is None else cv
        kind = StratifiedGroupKFold if grouped else StratifiedKFold
        base = kind(n_splits=n_splits)
        if classifier:
            return base
        # Quantile bins hold two rows per fold at least (`_continuous_strata`).
        # Below that the bins are thinner than the fold count and scikit-learn
        # refuses the split, talking about a "class" the caller never had.
        if n_rows < 2 * n_splits:
            raise ValueError(
                f"cv={cv!r} stratifies a continuous target on quantile bins of "
                f"y, which needs at least two rows per fold: {n_rows} row(s) "
                f"cannot fill {n_splits} folds. Use at most {n_rows // 2} "
                f"folds, or an unstratified splitter — "
                f"cv=KFold(n_splits={n_splits})."
            )
        return _ContinuousStratifiedSplitter(base)
    if not (hasattr(cv, "split") and hasattr(cv, "get_n_splits")):
        raise TypeError(
            f"cv must be None, an int fold count, or a scikit-learn "
            f"cross-validation splitter; got {type(cv).__name__}."
        )
    return cv


def _continuous_strata(y, n_splits: int, max_bins: int = 10) -> np.ndarray:
    """Quantile-bin a continuous target so stratified splitters can balance it.

    The bin count is capped so every bin holds at least `2 * n_splits` samples
    — enough for each fold to draw from every bin — and never exceeds
    `max_bins`. A target with few distinct values (an ordinal score) is used
    as-is, its own values becoming the labels.

    Every stratum ends up with at least `n_splits` members: a rare ordinal
    level, or ties sitting on a quantile edge, would otherwise leave one
    thinner than the fold count, and scikit-learn would then warn about a
    "class" a regression caller never had. The bins widen until that holds,
    down to a single stratum if the target is that degenerate. The caller
    guarantees at least `2 * n_splits` rows (`_resolve_splitter`).

    Args:
        y: The continuous target, one value per sample.
        n_splits: The fold count the strata will be split into.
        max_bins: The hard cap on the number of bins.

    Returns:
        Integer strata labels, one per sample.
    """
    y = np.asarray(y).ravel()
    n = y.shape[0]
    uniques = np.unique(y)
    n_bins = int(np.clip(n // (2 * n_splits), 2, max_bins))
    if uniques.size <= n_bins:
        labels = np.searchsorted(uniques, y)
        if np.bincount(labels).min() >= n_splits:
            return labels
    while n_bins > 1:
        edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
        labels = np.digitize(y, edges)
        if np.bincount(labels).min() >= n_splits:
            return labels
        n_bins -= 1
    return np.zeros(n, dtype=np.intp)


class _ContinuousStratifiedSplitter:
    """Adapt a stratified splitter to a continuous `y` via quantile bins.

    Exposes the scikit-learn splitter protocol (`split` / `get_n_splits`) so
    it slots into every code path that consumes `cv`; the binning happens at
    split time from whatever `y` the caller passes.

    It is not a `BaseCrossValidator`. Only `resolve_splits` consumes it, and
    it materializes the folds immediately, so the object itself never reaches
    scikit-learn's `check_cv`. Keep it that way, or make it a subclass.
    """

    def __init__(self, base):
        self.base = base

    @property
    def n_splits(self) -> int:
        return self.base.n_splits

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.base.get_n_splits(X, y, groups)

    def split(self, X, y=None, groups=None):
        strata = _continuous_strata(y, self.base.n_splits)
        yield from self.base.split(X, strata, groups=groups)

    def __repr__(self) -> str:
        return f"ContinuousStratified({self.base!r})"


def _as_indices(fold) -> np.ndarray:
    """Return one fold as an integer index array, accepting a boolean mask."""
    fold = np.asarray(fold)
    if fold.dtype == bool:
        return np.flatnonzero(fold)
    if fold.size == 0:
        return fold.astype(np.intp, copy=False)
    if not np.issubdtype(fold.dtype, np.integer):
        raise ValueError(
            f"Cross-validation splits must be integer indices or a boolean "
            f"mask; got dtype {fold.dtype}."
        )
    return fold


def _validate_partition(splits: list, *, n_rows: int) -> None:
    """Require the test folds to partition the rows: each row in exactly one."""
    if not splits:
        raise ValueError("The cross-validation splitter produced no folds.")
    assignments = np.concatenate([test for _, test in splits])
    counts = np.bincount(assignments, minlength=n_rows)
    repeated = int((counts > 1).sum())
    missing = int((counts == 0).sum())
    if repeated or missing:
        raise ValueError(
            f"Cross-validation test folds must partition the observations, so "
            f"each row appears in exactly one test fold: {repeated} row(s) "
            f"appear in more than one fold and {missing} row(s) appear in "
            f"none. Repeated, overlapping, and incomplete splitters (for "
            f"example ShuffleSplit or RepeatedKFold) are not accepted."
        )


# ---------------------------------------------------------------------------
# Whole-brain runner
# ---------------------------------------------------------------------------


def _fit_and_score_fold(X, y, pipe, scoring, train_idx, test_idx):
    """Fit one cross-validation fold and return its score and test predictions.

    A module-level function so `joblib` can ship it to a worker process
    directly. The scorer is rebuilt inside the worker because a scorer bound to
    an unfitted estimator does not survive the trip any more cheaply than the
    two arguments it is built from.
    """
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    fitted = clone(pipe).fit(X[train_idx], y[train_idx])
    scorer = check_scoring(fitted, scoring=scoring)
    score = float(scorer(fitted, X[test_idx], y[test_idx]))
    return score, np.asarray(fitted.predict(X[test_idx]))


def _run_whole_brain(bd, X, y, pipe, *, splits, scoring, classes, n_jobs) -> Predict:
    """A fit on all data for the map, then cross-validation for the scores.

    The canonical ``weight_map`` comes from a single fit on the full
    ``(X, y)``: one real estimator rather than an aggregation of K fold models,
    none of which the caller ever sees. That fit runs *first* so a pipeline
    whose coefficients cannot be projected back raises after one fit instead of
    after K + 1. Nothing observable is reordered — the folds are already
    materialized and the all-data fit does not depend on them.

    The cross-validation loop then produces honest scores and row-aligned
    out-of-fold predictions. ``n_jobs`` parallelizes that loop — folds are the
    outer independent work at this spatial scale — at the cost of one copy of
    the brain per worker.
    """
    from joblib import Parallel, delayed
    from sklearn.base import clone

    n_samples, n_voxels = X.shape

    estimator = clone(pipe).fit(X, y)
    weight_map_arr = _as_predict_map(back_project_weight_maps(estimator, n_voxels))

    if n_jobs == 1:
        fold_results = [
            _fit_and_score_fold(X, y, pipe, scoring, train_idx, test_idx)
            for train_idx, test_idx in splits
        ]
    else:
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_and_score_fold)(X, y, pipe, scoring, train_idx, test_idx)
            for train_idx, test_idx in splits
        )

    fold_scores = [score for score, _ in fold_results]
    fold_preds = [preds for _, preds in fold_results]
    fold_test_idx = [test_idx for _, test_idx in splits]

    fold_idx_array = np.empty(n_samples, dtype=int)
    for fold_idx, test_idx in enumerate(fold_test_idx):
        fold_idx_array[test_idx] = fold_idx

    # Assemble out-of-fold predictions with a dtype wide enough for every
    # fold — string class labels included (np.result_type widens e.g.
    # '<U4' vs '<U5'; float folds stay float). The folds partition the rows,
    # so every position is written exactly once.
    pred_dtype = (
        np.result_type(*(p.dtype for p in fold_preds))
        if fold_preds
        else np.dtype(float)
    )
    fold_predictions = np.zeros(n_samples, dtype=pred_dtype)
    for test_idx, preds in zip(fold_test_idx, fold_preds):
        fold_predictions[test_idx] = preds

    return Predict(
        spatial_scale="whole_brain",
        scoring=scoring,
        classes=getattr(estimator, "classes_", classes),
        predictions=fold_predictions,
        cv_folds=fold_idx_array,
        scores=np.asarray(fold_scores, dtype=float),
        estimator=estimator,
        weight_map=_to_braindata(bd, weight_map_arr),
    )


def _to_braindata(bd, arr):
    """Return one result map as a new, independently owned `BrainData`.

    The leading axis of a coefficient or score map is not the source
    observations, so the shared result policy clears the row metadata while
    copying the mask and masker state. Returns None if `arr` is None, which
    preserves "field not applicable" semantics.

    `Predict` deep-copies whatever it is handed, because a caller can construct
    one from a `BrainData` they still own. This map is therefore copied twice on
    the runner path; do not add a third copy here to "harden" it.
    """
    from .utils import _result_from_array

    if arr is None:
        return None
    return _result_from_array(bd, np.asarray(arr), rows="clear")


def _iter_split(cv, X, y, groups):
    """Iterate `cv.split`, passing `groups` only to splitters that accept it.

    The decision is made once from the signature, so a `TypeError` raised
    partway through a custom splitter's iteration surfaces to the caller
    instead of silently restarting the split.
    """
    try:
        accepts_groups = "groups" in inspect.signature(cv.split).parameters
    except (TypeError, ValueError):
        accepts_groups = True
    if accepts_groups:
        yield from cv.split(X, y, groups=groups)
    else:
        yield from cv.split(X, y)


# ---------------------------------------------------------------------------
# Weight-map extraction
# ---------------------------------------------------------------------------


def _as_predict_map(maps: np.ndarray) -> np.ndarray:
    """Shape back-projected coefficients the way `Predict.weight_map` requires.

    `nltools.algorithms.decoding` always returns ``(n_maps, n_features)``. The
    record wants one *unstacked* map for regression and binary classification
    and the stack for multiclass. `Predict` no longer re-checks that rule, so
    this is the only place it is enforced; every runner drops the leading axis
    here and nowhere else, and
    `test_braindata_prediction.py::TestWeightMapShapes` pins it.

    Args:
        maps: Back-projected coefficients, ``(n_maps, n_features)``.

    Returns:
        ndarray: ``(n_features,)`` when there is one map, else ``maps``.
    """
    return maps[0] if maps.shape[0] == 1 else maps


# ---------------------------------------------------------------------------
# Searchlight runner
# ---------------------------------------------------------------------------


def _score_sphere(X, y, pipe, splits, scoring, neighbor_indices) -> float:
    """Mean CV score for one searchlight sphere (NaN for degenerate/failed)."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    X_sphere = X[:, neighbor_indices]
    if X_sphere.shape[1] < 2:
        return np.nan
    try:
        scores = cross_val_score(clone(pipe), X_sphere, y, cv=splits, scoring=scoring)
        return float(np.mean(scores))
    except Exception:
        return np.nan


def _run_searchlight(
    bd, X, y, pipe, *, splits, scoring, classes, radius, n_jobs, progress_bar
) -> Predict:
    """Per-voxel-neighborhood CV decoding. Returns a Predict with one score_map.

    Local models fitted on overlapping neighborhoods have no common feature
    axis, so the result exposes no coefficient map, no fold assignments and no
    estimator — only the cross-fold mean score at each sphere center.
    """
    from joblib import Parallel, delayed

    from nltools.algorithms.neighborhoods import compute_searchlight_neighborhoods

    neighborhoods = compute_searchlight_neighborhoods(bd.mask, radius=radius)

    def decode_sphere(center_idx, neighbor_indices):
        return _score_sphere(X, y, pipe, splits, scoring, neighbor_indices)

    neighborhood_list = maybe_tqdm(
        list(neighborhoods.iter_neighborhoods()),
        progress_bar=progress_bar,
        desc="Searchlight",
        total=neighborhoods.n_voxels,
    )

    if n_jobs == 1:
        sphere_scores = [decode_sphere(c, n) for c, n in neighborhood_list]
    else:
        sphere_scores = Parallel(n_jobs=n_jobs)(
            delayed(decode_sphere)(c, n) for c, n in neighborhood_list
        )
    return Predict(
        spatial_scale="searchlight",
        scoring=scoring,
        classes=classes,
        score_map=_to_braindata(bd, np.asarray(sphere_scores, dtype=float)),
    )


# ---------------------------------------------------------------------------
# ROI runner
# ---------------------------------------------------------------------------


def _assemble_roi_weights(label_vec, unique_labels, per_roi) -> np.ndarray:
    """Write each parcel's coefficients into its voxels, NaN everywhere else.

    Args:
        label_vec: ``(n_voxels,)`` atlas label per in-mask voxel.
        unique_labels: The scored parcel labels, in score order.
        per_roi: One summary dict per parcel; ``coef`` is ``None`` for a parcel
            whose fit failed.

    Returns:
        ndarray: ``(n_voxels,)`` for one map, ``(n_classes, n_voxels)`` for
            multiclass. A parcel whose fit failed keeps NaN in its voxels.

    Raises:
        ValueError: If no parcel produced coefficients at all.
    """
    fitted = [r["coef"] for r in per_roi if r["coef"] is not None]
    if not fitted:
        raise ValueError(
            "No atlas parcel could be fitted, so no weight_map exists. Check "
            "that the atlas overlaps the mask and that every parcel has enough "
            "voxels and observations for the estimator."
        )
    n_maps = fitted[0].shape[0]
    weights = np.full((n_maps, label_vec.shape[0]), np.nan, dtype=float)
    for roi_label, summary in zip(unique_labels, per_roi):
        if summary["coef"] is not None:
            weights[:, label_vec == roi_label] = summary["coef"]
    return _as_predict_map(weights)


def _run_roi(
    bd, X, y, pipe, *, splits, scoring, classes, roi_mask, n_jobs, progress_bar
) -> Predict:
    """Per-parcel cross-validated decoding with an assembled voxel-space map.

    Returns a Predict with:

    - ``scores`` ``(n_folds, n_rois)`` — fold scores per parcel, in
      ``roi_labels`` order.
    - ``roi_labels`` ``(n_rois,)`` — atlas integer ids.
    - ``score_map`` — every voxel of parcel *i* set to that parcel's mean fold
      score (NaN outside parcels).
    - ``weight_map`` — per-parcel ``coef_`` from one all-data fit per parcel,
      written back into voxel space (NaN outside parcels).

    Assembly relies on each voxel belonging to exactly one parcel (the atlas is
    a label image, so this is structural). Cross-parcel weight magnitudes live
    on different feature distributions and are not directly comparable;
    within-parcel ranking is meaningful. The per-parcel estimators are internal:
    they are fitted to produce the map and are not exposed on the result.

    A pipeline whose coefficients cannot be projected back onto the parcel
    voxel axis raises `ValueError`, exactly as it does for whole-brain
    decoding. A parcel that simply *fails to fit* — too few voxels, one class
    in a training fold — is different: that parcel's ``scores`` column and its
    voxels in both maps come back NaN, and the rest of the atlas is still
    reported. Nothing warns and no field names the failed parcels; the matching
    NaN column in ``scores`` is how a caller identifies them. If *every* parcel
    fails, there is no map to assemble and the call raises.
    """
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    from nltools.data.braindata.analysis import _resolve_atlas_label_vec
    from nltools.data.results import _fold_mean

    _, label_vec, unique_labels = _resolve_atlas_label_vec(bd, roi_mask)

    n_folds = len(splits)

    def decode_roi(roi_label):
        """Cross-validate and refit one atlas parcel, returning its summary.

        The scorer is built here rather than closed over, so nothing scorer-
        shaped has to survive the worker boundary — the same rule
        `_fit_and_score_fold` follows for whole-brain folds.
        """
        failed = {"fold_scores": np.full(n_folds, np.nan), "coef": None}
        cols = label_vec == roi_label
        if not cols.any():
            return failed
        X_roi = X[:, cols]

        fold_scores = []
        try:
            for train_idx, test_idx in splits:
                fitted = clone(pipe).fit(X_roi[train_idx], y[train_idx])
                scorer = check_scoring(fitted, scoring=scoring)
                fold_scores.append(float(scorer(fitted, X_roi[test_idx], y[test_idx])))
            estimator = clone(pipe).fit(X_roi, y)
        except Exception:
            return failed

        # Outside the `try`: an unprojectable pipeline is a contract error for
        # the whole call, not a parcel that happened to fail.
        return {
            "fold_scores": np.asarray(fold_scores, dtype=float),
            "coef": back_project_weight_maps(estimator, int(cols.sum())),
        }

    iterator = maybe_tqdm(unique_labels, progress_bar=progress_bar, desc="ROI decoding")

    if n_jobs == 1:
        per_roi = [decode_roi(label) for label in iterator]
    else:
        per_roi = Parallel(n_jobs=n_jobs)(
            delayed(decode_roi)(label) for label in iterator
        )

    # Scores: (n_folds, n_rois)
    fold_scores_per_roi = np.vstack([r["fold_scores"] for r in per_roi]).T
    # The same reduction `Predict.mean_score` uses, so the painted map and the
    # reported summary cannot drift apart.
    mean_per_roi = _fold_mean(fold_scores_per_roi, axis=0)

    # score_map: every voxel carries the mean fold score of its parcel.
    score_arr = np.full(label_vec.shape, np.nan, dtype=float)
    for roi_label, parcel_score in zip(unique_labels, mean_per_roi):
        score_arr[label_vec == roi_label] = parcel_score

    # weight_map: assemble per-parcel coefficients back into voxel space. Each
    # voxel belongs to exactly one parcel, so every coefficient has one
    # destination. A parcel that failed to fit leaves NaN behind.
    weight_arr = _assemble_roi_weights(label_vec, unique_labels, per_roi)

    return Predict(
        spatial_scale="roi",
        scoring=scoring,
        classes=classes,
        scores=fold_scores_per_roi,
        weight_map=_to_braindata(bd, weight_arr),
        roi_labels=unique_labels.astype(np.int64),
        score_map=_to_braindata(bd, score_arr),
    )
