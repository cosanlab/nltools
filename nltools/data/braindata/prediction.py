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
    splits = resolve_splits(
        cv, X=bd.data, y=y, groups=groups, classifier=is_classifier(pipe)
    )

    X_data = bd.data  # (n_samples, n_voxels)

    if spatial_scale == "whole_brain":
        return _run_whole_brain(bd, X_data, y, pipe, splits, scoring)
    if spatial_scale == "searchlight":
        return _run_searchlight(
            bd, X_data, y, pipe, splits, scoring, radius, n_jobs, progress_bar
        )
    return _run_roi(
        bd, X_data, y, pipe, splits, scoring, roi_mask, n_jobs, progress_bar
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


def _run_whole_brain(bd, X, y, pipe, splits, scoring) -> Predict:
    """Cross-validated scoring + final fit on all data.

    The CV loop produces honest scores and out-of-fold predictions. Per-fold
    ``coef_`` vectors are stacked into ``fold_weight_maps`` for stability
    analysis but are *not* used for the canonical ``weight_map`` — that
    comes from a single fit on the full ``(X, y)`` (a real estimator, not
    an aggregation of K different fold models). The CV-mean of weights is
    one line away if anyone wants it: ``fold_weight_maps.data.mean(axis=0)``.
    """
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    n_samples = X.shape[0]
    n_voxels = X.shape[1]
    fold_scores: list[float] = []
    fold_idx_array = np.empty(n_samples, dtype=int)
    fold_weight_maps: list[np.ndarray | None] = []
    fold_preds: list[np.ndarray] = []
    fold_test_idx: list[np.ndarray] = []

    scorer = check_scoring(pipe, scoring=scoring)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fitted = clone(pipe).fit(X[train_idx], y[train_idx])
        score = scorer(fitted, X[test_idx], y[test_idx])
        fold_scores.append(float(score))
        fold_preds.append(np.asarray(fitted.predict(X[test_idx])))
        fold_test_idx.append(test_idx)
        fold_idx_array[test_idx] = fold_idx
        fold_weight_maps.append(_extract_weight_map(fitted, n_voxels))

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

    scores = np.asarray(fold_scores, dtype=float)
    _, fold_weight_maps_arr = _aggregate_weight_maps(
        fold_weight_maps, n_folds=len(fold_scores), n_voxels=n_voxels
    )

    # Always refit on all data — gives a single legitimate estimator and the
    # canonical weight_map for publication / interpretation. Cost: +1 fit.
    estimator = clone(pipe).fit(X, y)
    weight_map_arr = _extract_weight_map(estimator, n_voxels)

    return Predict(
        predictions=fold_predictions,
        scores=scores,
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        cv_folds=fold_idx_array,
        weight_map=_to_braindata(weight_map_arr, bd.mask),
        fold_weight_maps=_to_braindata(fold_weight_maps_arr, bd.mask),
        estimator=estimator,
    )


def _to_braindata(arr, mask):
    """Wrap a (n_voxels,) or (n_rows, n_voxels) array as BrainData with mask.

    Returns None if arr is None — preserves "field not applicable" semantics.
    """
    if arr is None:
        return None
    from nltools.data import BrainData

    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return BrainData(arr, mask=mask)


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

    For multi-class linear classifiers, returns the mean across classes.
    Back-projects through PCA when present. Returns None for non-linear
    models (no ``.coef_``) or when feature selection breaks back-projection
    (e.g., ``SelectKBest`` reduces feature count un-invertibly).

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
    # Collapse (n_classes, n_features) → (n_features,) by mean across classes.
    if coef.ndim == 2:
        coef = coef.mean(axis=0)
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


def _aggregate_weight_maps(
    per_fold: list[np.ndarray | None], n_folds: int, n_voxels: int
):
    """Aggregate per-fold weight maps into their voxelwise mean.

    Stacks maps into (n_folds, n_voxels) and averages to (n_voxels,).
    Returns (None, None) if any fold lacked a usable map.
    """
    if any(w is None for w in per_fold):
        return None, None
    stacked = np.vstack(per_fold)
    if stacked.shape != (n_folds, n_voxels):
        return None, None
    return stacked.mean(axis=0), stacked


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
    bd, X, y, pipe, splits, scoring, radius, n_jobs, progress_bar
) -> Predict:
    """Per-voxel-neighborhood CV decoding. Returns Predict with accuracy_map."""
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
        accuracies = [decode_sphere(c, n) for c, n in neighborhood_list]
    else:
        accuracies = Parallel(n_jobs=n_jobs)(
            delayed(decode_sphere)(c, n) for c, n in neighborhood_list
        )
    return Predict(
        accuracy_map=_to_braindata(np.asarray(accuracies, dtype=float), bd.mask)
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
    bd, X, y, pipe, splits, scoring, roi_mask, n_jobs, progress_bar
) -> Predict:
    """Per-ROI CV decoding with per-parcel weight maps.

    Returns Predict with:

    - ``scores`` ``(n_folds, n_rois)``, ``mean_score`` / ``std_score``
      ``(n_rois,)`` — fold scores per parcel and their cross-fold summary.
    - ``roi_labels`` ``(n_rois,)`` — atlas integer IDs in the same order.
    - ``accuracy_map`` BrainData ``(1, n_voxels)`` — every voxel inside parcel
      *i* set to that parcel's mean accuracy (others NaN).
    - ``weight_map`` BrainData ``(1, n_voxels)`` — per-parcel ``coef_`` vectors
      from one all-data fit per parcel, written back into voxel space. Voxels
      outside any parcel are NaN.
    - ``fold_weight_maps`` BrainData ``(n_folds, n_voxels)`` — same assembly
      per fold for stability analysis.
    - ``estimator`` ``dict[int, sklearn estimator]`` keyed by atlas label —
      the all-data fitted decoder for each parcel.

    Weight-map assembly relies on each voxel belonging to exactly one parcel
    (the atlas is a label image, so this is structural). Cross-parcel weight
    magnitudes live on different X distributions so are not directly
    comparable; within-parcel ranking is meaningful.

    If any parcel's estimator can't expose ``.coef_`` (non-linear model,
    ``SelectKBest`` pipeline), ``weight_map`` / ``fold_weight_maps`` /
    ``estimator`` are all None for the whole call (matches whole_brain's
    behavior for non-linear models).
    """
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    label_vec, unique_labels = _resolve_roi_labels(bd.mask, roi_mask)

    n_folds = len(splits)
    scorer = check_scoring(pipe, scoring=scoring)

    def decode_roi(roi_label):
        """Run cross-validation and an all-data refit for one atlas parcel.

        Captures per-fold scores and coefficients, then returns a tuple
        summarizing the parcel's result.
        """
        cols = label_vec == roi_label
        if not cols.any():
            return {
                "fold_scores": np.full(n_folds, np.nan),
                "fold_coefs": None,
                "estimator": None,
                "all_data_coef": None,
            }
        X_roi = X[:, cols]
        n_roi_voxels = int(cols.sum())

        fold_scores = []
        fold_coefs: list[np.ndarray | None] = []
        try:
            for train_idx, test_idx in splits:
                fitted = clone(pipe).fit(X_roi[train_idx], y[train_idx])
                fold_scores.append(float(scorer(fitted, X_roi[test_idx], y[test_idx])))
                fold_coefs.append(_extract_weight_map(fitted, n_roi_voxels, quiet=True))
            estimator = clone(pipe).fit(X_roi, y)
            all_data_coef = _extract_weight_map(estimator, n_roi_voxels, quiet=True)
        except Exception:
            return {
                "fold_scores": np.full(n_folds, np.nan),
                "fold_coefs": None,
                "estimator": None,
                "all_data_coef": None,
            }

        return {
            "fold_scores": np.asarray(fold_scores, dtype=float),
            "fold_coefs": fold_coefs,  # list of (n_roi_voxels,) arrays or Nones
            "estimator": estimator,
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
    mean_per_roi = np.nanmean(fold_scores_per_roi, axis=0)
    std_per_roi = np.nanstd(fold_scores_per_roi, axis=0)

    # accuracy_map: per-voxel mean accuracy for the parcel containing that voxel
    acc_arr = np.full(label_vec.shape, np.nan, dtype=float)
    for roi_label, acc in zip(unique_labels, mean_per_roi):
        acc_arr[label_vec == roi_label] = acc

    # weight_map / fold_weight_maps: assemble per-parcel coefs back to voxel
    # space. If any parcel couldn't expose coefs (non-linear, SelectKBest,
    # exception), set all weight fields to None and emit a single warning —
    # matches whole_brain's behavior of all-or-nothing.
    weight_extraction_failed = any(
        r["all_data_coef"] is None or r["fold_coefs"] is None for r in per_roi
    )
    if weight_extraction_failed:
        # Identify a representative failed parcel for the warning text
        warnings.warn(
            "Could not extract per-parcel coefficients for at least one "
            "ROI (non-linear model, SelectKBest in pipeline, or per-parcel "
            "fit error). Setting weight_map / fold_weight_maps / estimator "
            "to None for this call.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        weight_arr = None
        fold_weight_arr = None
        estimator_dict = None
    else:
        weight_arr = np.full(label_vec.shape, np.nan, dtype=float)
        fold_weight_arr = np.full((n_folds, label_vec.shape[0]), np.nan, dtype=float)
        for roi_label, r in zip(unique_labels, per_roi):
            cols = label_vec == roi_label
            weight_arr[cols] = r["all_data_coef"]
            for f_idx in range(n_folds):
                fold_weight_arr[f_idx, cols] = r["fold_coefs"][f_idx]
        estimator_dict = {
            int(label): r["estimator"] for label, r in zip(unique_labels, per_roi)
        }

    return Predict(
        scores=fold_scores_per_roi,
        mean_score=mean_per_roi,
        std_score=std_per_roi,
        roi_labels=unique_labels.astype(np.int64),
        accuracy_map=_to_braindata(acc_arr, bd.mask),
        weight_map=_to_braindata(weight_arr, bd.mask),
        fold_weight_maps=_to_braindata(fold_weight_arr, bd.mask),
        estimator=estimator_dict,
    )
