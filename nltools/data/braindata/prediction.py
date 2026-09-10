"""BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: `predict`. It resolves exactly one mode, validates every
argument for that mode, and returns either a new `BrainData` (fitted-model
prediction) or a frozen `Predict` record (MVPA). Nothing is attached to the
source object.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np

from nltools.data.results import Predict
from nltools.utils import find_stack_level, maybe_tqdm


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


def _is_default(value, default) -> bool:
    """Compare an argument against its documented default without ambiguity."""
    if isinstance(value, np.ndarray):
        return False
    if default is None:
        return value is None
    if isinstance(default, float):
        return isinstance(value, (int, float)) and float(value) == default
    return type(value) is type(default) and value == default


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

    pipe = build_pipeline(estimator)
    classifier = is_classifier(pipe)
    splits = resolve_splits(cv, X=bd.data, y=y, groups=groups, classifier=classifier)
    classes = np.unique(y) if classifier else None

    X_data = bd.data  # (n_samples, n_voxels)

    if spatial_scale == "whole_brain":
        return _run_whole_brain(bd, X_data, y, pipe, splits, scoring, classes)
    if spatial_scale == "searchlight":
        return _run_searchlight(
            bd, X_data, y, pipe, splits, scoring, classes, radius, n_jobs, progress_bar
        )
    return _run_roi(
        bd, X_data, y, pipe, splits, scoring, classes, roi_mask, n_jobs, progress_bar
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


def build_pipeline(estimator: Any):
    """Build the per-fold pipeline for `estimator`.

    A built-in shortcut selects a predefined pipeline that standardizes
    features inside each fold before fitting. A caller-supplied estimator or
    `Pipeline` is used exactly as given — MVPA adds, removes, and
    reconfigures nothing.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    resolved = resolve_estimator(estimator)
    if isinstance(estimator, str):
        return make_pipeline(StandardScaler(), resolved)
    return resolved


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def resolve_splits(cv, *, X, y, groups, classifier: bool) -> list:
    """Resolve `cv` into materialized train/test splits and check the partition.

    Materializing once means every runner — and every parallel worker — sees
    the same folds, and it lets the partition rule be checked before any model
    is fitted.
    """
    splitter = _resolve_splitter(cv, classifier=classifier)
    splits = [
        (_as_indices(train), _as_indices(test))
        for train, test in _iter_split(splitter, X, y, groups)
    ]
    _validate_partition(splits, n_rows=len(y))
    return splits


def _resolve_splitter(cv, *, classifier: bool):
    """Turn a `cv` spec into a scikit-learn splitter following sklearn's grammar.

    This deliberately does not call the public `nltools.cross_validation.resolve_cv`.
    That helper keeps the `'loo'`/`'logo'` names, promotes an int to a
    group-aware splitter, and can shuffle; the prediction spec removes all
    three from `predict` (see the cross-validation paragraph of
    `docs/development/specs/braindata.md`), so the two rules have genuinely
    different semantics.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    if isinstance(cv, str):
        raise ValueError(
            f"cv={cv!r} is not accepted: the 'loo' and 'logo' aliases were "
            f"removed from predict. Pass the splitter itself — "
            f"LeaveOneOut() or LeaveOneGroupOut() with groups= — an int fold "
            f"count, or None for a deterministic five-fold split."
        )
    if cv is None or (isinstance(cv, int) and not isinstance(cv, bool)):
        n_splits = 5 if cv is None else cv
        return (
            StratifiedKFold(n_splits=n_splits)
            if classifier
            else KFold(n_splits=n_splits)
        )
    if not (hasattr(cv, "split") and hasattr(cv, "get_n_splits")):
        raise TypeError(
            f"cv must be None, an int fold count, or a scikit-learn "
            f"cross-validation splitter; got {type(cv).__name__}."
        )
    return cv


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


def _run_whole_brain(bd, X, y, pipe, splits, scoring, classes) -> Predict:
    """Cross-validated scoring and out-of-fold predictions, then a fit on all data.

    The cross-validation loop produces honest scores and row-aligned
    out-of-fold predictions. The canonical ``weight_map`` comes from a single
    fit on the full ``(X, y)`` — one real estimator rather than an aggregation
    of K fold models, none of which the caller ever sees.
    """
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    n_samples, n_voxels = X.shape
    fold_scores: list[float] = []
    fold_idx_array = np.empty(n_samples, dtype=int)
    fold_preds: list[np.ndarray] = []
    fold_test_idx: list[np.ndarray] = []

    scorer = check_scoring(pipe, scoring=scoring)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fitted = clone(pipe).fit(X[train_idx], y[train_idx])
        fold_scores.append(float(scorer(fitted, X[test_idx], y[test_idx])))
        fold_preds.append(np.asarray(fitted.predict(X[test_idx])))
        fold_test_idx.append(test_idx)
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

    # Always refit on all data — one legitimate estimator, and the canonical
    # weight_map for publication and interpretation. Cost: one extra fit.
    estimator = clone(pipe).fit(X, y)
    weight_map_arr = _extract_weight_map(estimator, n_voxels)

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
# Weight-map extraction (with optional PCA back-projection)
# ---------------------------------------------------------------------------


def _extract_weight_map(
    fitted_pipe, n_features: int, *, quiet: bool = False
) -> np.ndarray | None:
    """Extract a one-dimensional coefficient vector from a fitted pipeline.

    The vector uses the local feature width (n_voxels for whole_brain, sphere size for searchlight,
    parcel size for ROI).

    Back-projects through PCA when present. Returns None for non-linear models
    (no ``.coef_``), for multiclass coefficients (one map per class needs the
    back-projection work; averaging across classes is not a valid map), or when
    feature selection breaks back-projection (e.g., ``SelectKBest`` reduces
    feature count un-invertibly).

    ``quiet=True`` silences the no-``.coef_`` warning — used by ROI/searchlight
    runners that aggregate a single warning at the runner level so a
    non-linear-model call doesn't emit one warning per parcel/sphere.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    # Unwrap Pipeline / make_pipeline to find the final estimator and any
    # preceding PCA.
    if isinstance(fitted_pipe, Pipeline):
        named = list(fitted_pipe.named_steps.values())
    else:
        named = [fitted_pipe]
    final_est = named[-1]
    pca_step = next((step for step in named[:-1] if isinstance(step, PCA)), None)

    coef = getattr(final_est, "coef_", None)
    if coef is None:
        if not quiet:
            warnings.warn(
                f"{type(final_est).__name__} has no .coef_ attribute; "
                "weight_map is unavailable for non-linear models. Pass a "
                "linear estimator= — 'linear_svc', 'logistic_regression', "
                "'linear_discriminant_analysis', 'ridge_classifier', 'ridge', "
                "'lasso', 'linear_svr' — or compute permutation importances "
                "directly.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        return None

    coef = np.asarray(coef)
    # A binary classifier or a regressor exposes one coefficient row.
    if coef.ndim == 2 and coef.shape[0] == 1:
        coef = coef.ravel()
    elif coef.ndim == 2:
        if not quiet:
            warnings.warn(
                f"{type(final_est).__name__} fitted {coef.shape[0]} classes, and "
                "one coefficient map per class is not available yet, so "
                "weight_map is None for this call. Averaging coefficients across "
                "classes would not describe any fitted decision boundary. The "
                "cross-validated scores and out-of-fold predictions are "
                "unaffected.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        return None
    elif coef.ndim != 1:
        return None

    # Back-project through PCA if present: pca.components_.T @ coef
    if pca_step is not None:
        coef = pca_step.components_.T @ coef

    if coef.shape[0] != n_features:
        # Some estimators (e.g., SelectKBest pipelines) reduce feature count
        # in ways we can't trivially reverse — bail out.
        return None
    return coef


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
    bd, X, y, pipe, splits, scoring, classes, radius, n_jobs, progress_bar
) -> Predict:
    """Per-voxel-neighborhood CV decoding. Returns a Predict with one score_map.

    Local models fitted on overlapping neighborhoods have no common feature
    axis, so the result exposes no coefficient map, no fold assignments and no
    estimator — only the cross-fold mean score at each sphere center.
    """
    from joblib import Parallel, delayed

    from .neighborhoods import compute_searchlight_neighborhoods

    neighborhoods = compute_searchlight_neighborhoods(
        bd.mask, radius_mm=radius, use_cache=True
    )

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


def _resolve_roi_labels(brain_mask, roi_mask) -> tuple[np.ndarray, np.ndarray]:
    """Resolve an atlas image into per-voxel labels on ``brain_mask``.

    Loads a path, resamples (nearest) into the brain mask's grid when shapes
    or affines differ, and returns ``(label_vec, unique_labels)`` where
    ``label_vec`` is the ``(n_voxels,)`` int atlas label per in-mask voxel
    and ``unique_labels`` the sorted non-zero labels used by ``_run_roi``.
    """
    from pathlib import Path

    import nibabel as nib
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask

    if roi_mask is None:
        raise ValueError("roi_mask required for spatial_scale='roi'")

    if isinstance(roi_mask, (str, Path)):
        roi_mask = nib.load(roi_mask)

    if roi_mask.shape != brain_mask.shape or not np.allclose(
        roi_mask.affine, brain_mask.affine
    ):
        roi_mask = resample_to_img(
            roi_mask,
            brain_mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    label_vec = apply_mask(roi_mask, brain_mask).astype(np.int64)
    unique_labels = np.unique(label_vec)
    unique_labels = unique_labels[unique_labels != 0]
    return label_vec, unique_labels


def _run_roi(
    bd, X, y, pipe, splits, scoring, classes, roi_mask, n_jobs, progress_bar
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

    If any parcel's estimator cannot expose ``coef_`` (a non-linear model,
    feature selection in the pipeline, or a per-parcel fit error),
    ``weight_map`` is None for the whole call, matching whole-brain's behavior.
    """
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    from nltools.data.results import _fold_mean

    label_vec, unique_labels = _resolve_roi_labels(bd.mask, roi_mask)

    n_folds = len(splits)
    scorer = check_scoring(pipe, scoring=scoring)

    def decode_roi(roi_label):
        """Cross-validate and refit one atlas parcel, returning its summary."""
        failed = {"fold_scores": np.full(n_folds, np.nan), "all_data_coef": None}
        cols = label_vec == roi_label
        if not cols.any():
            return failed
        X_roi = X[:, cols]
        n_roi_voxels = int(cols.sum())

        fold_scores = []
        try:
            for train_idx, test_idx in splits:
                fitted = clone(pipe).fit(X_roi[train_idx], y[train_idx])
                fold_scores.append(float(scorer(fitted, X_roi[test_idx], y[test_idx])))
            estimator = clone(pipe).fit(X_roi, y)
            all_data_coef = _extract_weight_map(estimator, n_roi_voxels, quiet=True)
        except Exception:
            return failed

        return {
            "fold_scores": np.asarray(fold_scores, dtype=float),
            "all_data_coef": all_data_coef,
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

    # weight_map: assemble per-parcel coefficients back into voxel space. If
    # any parcel could not expose them, drop the map for the whole call and
    # warn once — matching whole_brain's all-or-nothing behavior.
    weight_arr = None
    if any(r["all_data_coef"] is None for r in per_roi):
        warnings.warn(
            "Could not extract per-parcel coefficients for at least one ROI "
            "(non-linear model, feature selection in the pipeline, or a "
            "per-parcel fit error), so weight_map is None for this call.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
    else:
        weight_arr = np.full(label_vec.shape, np.nan, dtype=float)
        for roi_label, r in zip(unique_labels, per_roi):
            weight_arr[label_vec == roi_label] = r["all_data_coef"]

    return Predict(
        spatial_scale="roi",
        scoring=scoring,
        classes=classes,
        scores=fold_scores_per_roi,
        weight_map=_to_braindata(bd, weight_arr),
        roi_labels=unique_labels.astype(np.int64),
        score_map=_to_braindata(bd, score_arr),
    )
