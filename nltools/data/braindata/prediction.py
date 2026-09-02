"""BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: `predict`. Returns `Predict`
with fields populated based on dispatch. Mirrors `BrainData.fit` /
`Fit` patterns: frozen result dataclass, ``inplace=True`` mutates
self with attributes, ``inplace=False`` returns the dataclass.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from nltools.data.fitresults import Predict
from nltools.utils import maybe_tqdm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def predict(
    bd,
    *,
    y=None,
    X=None,
    spatial_scale: str = "whole_brain",
    model: Any = "svm",
    cv: int = 5,
    standardize: bool = True,
    reduce: str | None = None,
    n_components: int | None = None,
    scoring: str = "auto",
    groups=None,
    roi_mask=None,
    radius_mm: float = 10.0,
    inplace: bool = False,
    n_jobs: int = 1,
    random_state: int | None = None,
    progress_bar: bool = False,
):
    """Dispatch BrainData prediction to timeseries encoding or MVPA decoding.

    Implements `BrainData.predict`. See the class docstring for full parameter
    documentation.
    """
    if X is not None and y is not None:
        raise ValueError(
            "Cannot specify both X and y. Use X for timeseries prediction "
            "or y for MVPA decoding."
        )

    if X is None:
        y = _resolve_stored_y(bd, y)
        groups = _resolve_stored_groups(bd, groups)

    if y is not None:
        return predict_mvpa(
            bd,
            y=y,
            spatial_scale=spatial_scale,
            model=model,
            cv=cv,
            standardize=standardize,
            reduce=reduce,
            n_components=n_components,
            scoring=scoring,
            groups=groups,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            inplace=inplace,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )
    return predict_timeseries(bd, X=X)


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
        dispatcher can fall through to timeseries prediction — unless the
        object also carries a fitted encoding model, in which case both
        readings are possible and we refuse to guess.
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

    if hasattr(bd, "model_"):
        raise ValueError(
            "predict() is ambiguous: this BrainData has both a fitted "
            "encoding model and a stored .Y frame. Pass y=... (or y='column') "
            "to decode, or X=... to predict a timeseries."
        )
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
    """
    from nltools.data import BrainData
    from nltools.models import Glm

    from .utils import shallow_copy

    if not hasattr(bd, "model_"):
        raise ValueError(
            "Must call fit() before predict() for timeseries prediction. "
            "Example: brain_data.fit(model='ridge', X=features)"
        )
    if not bd.model_.is_fitted_:
        raise ValueError("Model is not fitted")

    using_training_data = X is None
    if using_training_data:
        if not hasattr(bd, "X_"):
            raise ValueError("No training data stored on BrainData")
        X = bd.X_

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.ndim}D")

    if hasattr(bd.model_, "n_features_in_"):
        if X.shape[1] != bd.model_.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{bd.model_.n_features_in_} features"
            )
    elif hasattr(bd.model_, "design_matrices_") and bd.model_.design_matrices_:
        expected = bd.model_.design_matrices_[0].shape[1]
        if X.shape[1] != expected:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with "
                f"{expected} features"
            )

    if isinstance(bd.model_, Glm) and using_training_data:
        # Training-data fitted values come back as nilearn Nifti images; remask
        # to the BrainData array space. New-design prediction (X given) goes
        # through model.predict(X) = X @ coef_, identical to the Ridge path.
        y_pred_list = bd.model_.predict()
        y_pred = BrainData(y_pred_list, mask=bd.mask).data
    else:
        y_pred = bd.model_.predict(X)

    predictions = shallow_copy(bd)
    predictions.data = y_pred
    return predictions


# ---------------------------------------------------------------------------
# MVPA decoding
# ---------------------------------------------------------------------------


VALID_SPATIAL_SCALES = {"whole_brain", "searchlight", "roi"}


def predict_mvpa(
    bd,
    *,
    y,
    spatial_scale: str,
    model: Any,
    cv,
    standardize: bool,
    reduce: str | None,
    n_components: int | None,
    scoring: str,
    groups,
    roi_mask,
    radius_mm: float,
    inplace: bool,
    n_jobs: int,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> Predict | Any:
    """Run cross-validated decoding on a `BrainData`.

    Returns a `Predict` result, or `bd` itself when ``inplace=True``.
    """
    from sklearn.base import is_classifier

    from nltools.cross_validation import resolve_cv

    if spatial_scale not in VALID_SPATIAL_SCALES:
        raise ValueError(
            f"Invalid spatial_scale: {spatial_scale!r}. "
            f"Must be one of {sorted(VALID_SPATIAL_SCALES)}"
        )
    if reduce is not None and reduce != "pca":
        raise ValueError(f"Unknown reduce: {reduce!r}. Supported: 'pca' or None.")

    resolved_model = resolve_model(model)
    classifier = is_classifier(resolved_model)
    scoring = resolve_scoring(scoring, classifier)

    # Resolve CV — int specs shuffle here (single-subject rows are often
    # ordered by condition, so unshuffled contiguous folds would be
    # degenerate); the group variants derive folds from ``groups`` instead.
    cv_splitter = resolve_cv(
        cv,
        groups=groups,
        classifier=classifier,
        shuffle=True,
        random_state=random_state,
    )

    y = np.asarray(y)
    if y.shape[0] != bd.shape[0]:
        raise ValueError(
            f"y has {y.shape[0]} samples but data has {bd.shape[0]} samples"
        )

    standardize = _resolve_standardize_for_model(resolved_model, standardize)
    pipe = build_pipeline(resolved_model, standardize, reduce, n_components)
    X_data = bd.data  # (n_samples, n_voxels)

    if spatial_scale == "whole_brain":
        result = _run_whole_brain(bd, X_data, y, pipe, cv_splitter, groups, scoring)
    elif spatial_scale == "searchlight":
        result = _run_searchlight(
            bd,
            X_data,
            y,
            pipe,
            cv_splitter,
            groups,
            scoring,
            radius_mm,
            n_jobs,
            progress_bar,
        )
    else:  # spatial_scale == 'roi'
        result = _run_roi(
            bd,
            X_data,
            y,
            pipe,
            cv_splitter,
            groups,
            scoring,
            roi_mask,
            n_jobs,
            progress_bar,
        )

    if inplace:
        for fname in result.available():
            setattr(bd, f"predict_{fname}", getattr(result, fname))
        return bd
    return result


def _resolve_standardize_for_model(resolved_model, standardize: bool) -> bool:
    """Resolve standardization for user-provided scikit-learn pipelines.

    Auto-default ``standardize=False`` when the user passes a sklearn
    ``Pipeline`` as ``model=``. Pipelines are an explicit "I'm taking control
    of preprocessing" signal — silently wrapping another StandardScaler around
    them double-scales features. Users can opt back in with ``standardize=True``
    explicitly.
    """
    from sklearn.pipeline import Pipeline

    if standardize and isinstance(resolved_model, Pipeline):
        warnings.warn(
            "Detected sklearn Pipeline as model — defaulting standardize=False "
            "to avoid wrapping another StandardScaler around your pipeline. "
            "Pass standardize=True explicitly to override.",
            UserWarning,
            stacklevel=4,
        )
        return False
    return standardize


# ---------------------------------------------------------------------------
# Helpers — model / scoring / pipeline construction
# ---------------------------------------------------------------------------


def resolve_model(model: Any):
    """Resolve a string shortcut or pass through a sklearn estimator."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import Lasso, LogisticRegression, Ridge, RidgeClassifier
    from sklearn.svm import SVR, LinearSVC

    shortcuts = {
        # classification
        "svm": lambda: LinearSVC(dual="auto", max_iter=10000),
        "logistic": lambda: LogisticRegression(max_iter=1000),
        "lda": lambda: LinearDiscriminantAnalysis(),
        "ridge_classifier": lambda: RidgeClassifier(),
        # regression
        "ridge": lambda: Ridge(),
        "lasso": lambda: Lasso(),
        "svr": lambda: SVR(),
    }

    if isinstance(model, str):
        if model not in shortcuts:
            raise ValueError(
                f"Unknown model: {model!r}. Valid shortcuts: "
                f"{sorted(shortcuts)}, or pass any sklearn estimator."
            )
        return shortcuts[model]()

    if not (hasattr(model, "fit") and hasattr(model, "predict")):
        raise TypeError(
            f"model must be a string shortcut or an object with fit/predict; "
            f"got {type(model).__name__}"
        )
    return model


def _serialize_model_spec(model: Any) -> dict:
    """Build a JSON-able refit spec for a ``model=`` argument.

    Predict bundles persist the *ingredients* of a decoding run rather than
    a pickled estimator; this is the model half of that contract. Shapes:

    - string shortcut → ``{'kind': 'shortcut', 'name': 'svm',
      'refittable': True}``;
    - estimator whose shallow ``get_params()`` survive JSON →
      ``{'kind': 'estimator', 'class': 'sklearn.svm._classes.SVC',
      'params': {...}, 'refittable': True}``;
    - anything else (nested estimators, callables, array params) →
      ``{'kind': 'repr', 'repr': repr(model), 'refittable': False}`` — the
      bundle is explicitly marked non-refittable rather than pretending a
      repr string is a spec.

    ``_model_from_spec`` is the inverse.
    """
    import json

    if isinstance(model, str):
        return {"kind": "shortcut", "name": model, "refittable": True}
    if hasattr(model, "get_params"):
        try:
            params = model.get_params(deep=False)
            json.dumps(params)
        except (TypeError, ValueError):
            pass
        else:
            cls = type(model)
            return {
                "kind": "estimator",
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "params": params,
                "refittable": True,
            }
    return {"kind": "repr", "repr": repr(model), "refittable": False}


def _model_from_spec(spec: dict | str):
    """Reconstruct an unfitted estimator from a ``_serialize_model_spec`` dict.

    A bare string (legacy spec form) is treated as a shortcut name. Raises
    ``ValueError`` for specs marked non-refittable.
    """
    import importlib

    if isinstance(spec, str):
        return resolve_model(spec)
    kind = spec.get("kind")
    if kind == "shortcut":
        return resolve_model(spec["name"])
    if kind == "estimator":
        module_path, _, cls_name = spec["class"].rpartition(".")
        cls = getattr(importlib.import_module(module_path), cls_name)
        return cls(**spec["params"])
    raise ValueError(
        "model spec is not refittable (its params could not be serialized); "
        f"stored repr: {spec.get('repr', '<missing>')}. Rebuild the estimator "
        "by hand to refit."
    )


def resolve_scoring(scoring: str, classifier: bool) -> str:
    """Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor)."""
    if scoring == "auto":
        return "accuracy" if classifier else "r2"
    return scoring


def build_pipeline(model, standardize: bool, reduce: str | None, n_components):
    """Build a per-fold scikit-learn preprocessing and model pipeline.

    The pipeline contains an optional StandardScaler, optional PCA, and the
    model. If only the model is needed, returns the model itself.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    steps = []
    if standardize:
        steps.append(StandardScaler())
    if reduce == "pca":
        steps.append(PCA(n_components=n_components))
    steps.append(model)
    return make_pipeline(*steps) if len(steps) > 1 else model


# ---------------------------------------------------------------------------
# Whole-brain runner
# ---------------------------------------------------------------------------


def _run_whole_brain(bd, X, y, pipe, cv, groups, scoring) -> Predict:
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
    fold_idx_array = np.full(n_samples, -1, dtype=int)
    fold_weight_maps: list[np.ndarray | None] = []
    fold_preds: list[np.ndarray] = []
    fold_test_idx: list[np.ndarray] = []

    scorer = check_scoring(pipe, scoring=scoring)

    for fold_idx, (train_idx, test_idx) in enumerate(_iter_split(cv, X, y, groups)):
        fitted = clone(pipe).fit(X[train_idx], y[train_idx])
        score = scorer(fitted, X[test_idx], y[test_idx])
        fold_scores.append(float(score))
        fold_preds.append(np.asarray(fitted.predict(X[test_idx])))
        fold_test_idx.append(np.asarray(test_idx))
        fold_idx_array[test_idx] = fold_idx
        fold_weight_maps.append(_extract_weight_map(fitted, n_voxels))

    # Assemble out-of-fold predictions with a dtype wide enough for every
    # fold — string class labels included (np.result_type widens e.g.
    # '<U4' vs '<U5'; float folds stay float). Samples never in a test fold
    # (possible with e.g. ShuffleSplit) keep the dtype's zero value and are
    # identifiable via cv_folds == -1.
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
    """Wrap cv.split to handle splitters that ignore/require groups."""
    try:
        yield from cv.split(X, y, groups=groups)
    except TypeError:
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
                "weight_map is unavailable for non-linear models. Use a linear "
                "model ('svm', 'logistic', 'ridge_classifier', 'lda', 'ridge', "
                "'lasso') or compute permutation importances directly.",
                UserWarning,
                stacklevel=4,
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


def _score_sphere(X, y, pipe, cv, groups, scoring, neighbor_indices) -> float:
    """Mean CV score for one searchlight sphere (NaN for degenerate/failed)."""
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    X_sphere = X[:, neighbor_indices]
    if X_sphere.shape[1] < 2:
        return np.nan
    try:
        scores = cross_val_score(
            clone(pipe), X_sphere, y, cv=cv, groups=groups, scoring=scoring
        )
        return float(np.mean(scores))
    except Exception:
        return np.nan


def _run_searchlight(
    bd, X, y, pipe, cv, groups, scoring, radius_mm, n_jobs, progress_bar
) -> Predict:
    """Per-voxel-neighborhood CV decoding. Returns Predict with accuracy_map."""
    from joblib import Parallel, delayed

    from .neighborhoods import compute_searchlight_neighborhoods

    neighborhoods = compute_searchlight_neighborhoods(
        bd.mask, radius_mm=radius_mm, use_cache=True
    )

    def decode_sphere(center_idx, neighbor_indices):
        return _score_sphere(X, y, pipe, cv, groups, scoring, neighbor_indices)

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
    and ``unique_labels`` the sorted non-zero labels. Shared by ``_run_roi``
    and ``BrainCollection.predict_group``'s permutation null.
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
    bd, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, progress_bar
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

    n_folds = cv.get_n_splits(X, y, groups=groups)
    # Pre-compute split indices once so workers see the same splits; cv objects
    # with random_state already control this, but materializing makes the
    # per-parcel parallel loop deterministic and pickle-safe.
    split_indices = list(_iter_split(cv, X, y, groups))
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
            for train_idx, test_idx in split_indices:
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
            stacklevel=4,
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


# ---------------------------------------------------------------------------
# Score-only CV cores — used by BrainCollection.predict_group's permutation
# null so null iterations pay for scoring only (no all-data refit, no
# weight-map extraction, no map assembly). Each mirrors its runner's fold
# loop exactly so null values match a full re-run for the same labels.
# ---------------------------------------------------------------------------


def _cv_mean_score(X, y, pipe, cv, groups, scoring) -> float:
    """Mean CV score only — the scoring core of ``_run_whole_brain``."""
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    scorer = check_scoring(pipe, scoring=scoring)
    fold_scores = [
        float(
            scorer(
                clone(pipe).fit(X[train_idx], y[train_idx]), X[test_idx], y[test_idx]
            )
        )
        for train_idx, test_idx in _iter_split(cv, X, y, groups)
    ]
    return float(np.mean(fold_scores))


def _cv_roi_mean_scores(
    X, y, pipe, cv, groups, scoring, label_vec, unique_labels
) -> np.ndarray:
    """Per-parcel mean CV scores only — the scoring core of ``_run_roi``.

    Returns ``(n_rois,)`` with NaN for empty or failing parcels, matching
    ``_run_roi``'s per-parcel error semantics.
    """
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    split_indices = list(_iter_split(cv, X, y, groups))
    scorer = check_scoring(pipe, scoring=scoring)
    out = np.full(len(unique_labels), np.nan, dtype=float)
    for k, roi_label in enumerate(unique_labels):
        cols = label_vec == roi_label
        if not cols.any():
            continue
        X_roi = X[:, cols]
        try:
            fold_scores = [
                float(
                    scorer(
                        clone(pipe).fit(X_roi[train_idx], y[train_idx]),
                        X_roi[test_idx],
                        y[test_idx],
                    )
                )
                for train_idx, test_idx in split_indices
            ]
        except Exception:
            continue
        out[k] = float(np.nanmean(fold_scores))
    return out


def _cv_searchlight_scores(
    X, y, pipe, cv, groups, scoring, neighborhood_list
) -> np.ndarray:
    """Per-voxel sphere scores only — the scoring core of ``_run_searchlight``.

    ``neighborhood_list`` is ``[(center_idx, neighbor_indices), ...]`` in
    mask-voxel order; returns ``(n_voxels,)`` with NaN for degenerate or
    failing spheres.
    """
    return np.asarray(
        [
            _score_sphere(X, y, pipe, cv, groups, scoring, neighbor_indices)
            for _, neighbor_indices in neighborhood_list
        ],
        dtype=float,
    )
