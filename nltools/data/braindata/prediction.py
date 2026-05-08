"""BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: :func:`predict`. Returns :class:`nltools.data.fitresults.Predict`
with fields populated based on dispatch. Mirrors :meth:`BrainData.fit` /
:class:`Fit` patterns: frozen result dataclass, ``inplace=True`` mutates
self with attributes, ``inplace=False`` returns the dataclass.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from nltools.data.fitresults import Predict


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def predict(
    bd,
    *,
    y=None,
    X=None,
    method: str = "whole_brain",
    model: Any = "svm",
    cv: int = 5,
    standardize: bool = True,
    reduce: str | None = None,
    n_components: int | None = None,
    scoring: str = "auto",
    refit: bool = False,
    groups=None,
    roi_mask=None,
    radius_mm: float = 10.0,
    inplace: bool = False,
    n_jobs: int = 1,
    progress_bar: bool = False,
):
    """Implementation of :meth:`BrainData.predict`. See class docstring for
    full parameter documentation.
    """
    if X is not None and y is not None:
        raise ValueError(
            "Cannot specify both X and y. Use X for timeseries prediction "
            "or y for MVPA decoding."
        )

    if y is not None:
        return predict_mvpa(
            bd,
            y=y,
            method=method,
            model=model,
            cv=cv,
            standardize=standardize,
            reduce=reduce,
            n_components=n_components,
            scoring=scoring,
            refit=refit,
            groups=groups,
            roi_mask=roi_mask,
            radius_mm=radius_mm,
            inplace=inplace,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
    return predict_timeseries(bd, X=X)


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

    if isinstance(bd.model_, Glm):
        if using_training_data:
            y_pred_list = bd.model_.predict()
            y_pred = BrainData(y_pred_list, mask=bd.mask).data
        else:
            raise NotImplementedError(
                "Prediction with new design matrix not yet implemented for GLM."
            )
    else:
        y_pred = bd.model_.predict(X)

    predictions = shallow_copy(bd)
    predictions.data = y_pred
    return predictions


# ---------------------------------------------------------------------------
# MVPA decoding
# ---------------------------------------------------------------------------


VALID_METHODS = {"whole_brain", "searchlight", "roi"}


def predict_mvpa(
    bd,
    *,
    y,
    method: str,
    model: Any,
    cv,
    standardize: bool,
    reduce: str | None,
    n_components: int | None,
    scoring: str,
    refit: bool,
    groups,
    roi_mask,
    radius_mm: float,
    inplace: bool,
    n_jobs: int,
    progress_bar: bool,
) -> Predict | Any:
    """Cross-validated decoding. Returns Predict (or self if inplace=True)."""
    from sklearn.base import is_classifier
    from sklearn.model_selection import KFold, StratifiedKFold

    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method: {method!r}. Must be one of {sorted(VALID_METHODS)}"
        )
    if reduce is not None and reduce != "pca":
        raise ValueError(f"Unknown reduce: {reduce!r}. Supported: 'pca' or None.")

    resolved_model = resolve_model(model)
    classifier = is_classifier(resolved_model)
    scoring = resolve_scoring(scoring, classifier)

    # Resolve CV
    if isinstance(cv, int):
        cv_splitter = (
            StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            if classifier
            else KFold(n_splits=cv, shuffle=True, random_state=42)
        )
    else:
        cv_splitter = cv

    y = np.asarray(y)
    if y.shape[0] != bd.shape[0]:
        raise ValueError(
            f"y has {y.shape[0]} samples but data has {bd.shape[0]} samples"
        )

    standardize = _resolve_standardize_for_model(resolved_model, standardize)
    pipe = build_pipeline(resolved_model, standardize, reduce, n_components)
    X_data = bd.data  # (n_samples, n_voxels)

    if method == "whole_brain":
        result = _run_whole_brain(
            bd, X_data, y, pipe, cv_splitter, groups, scoring, refit
        )
    elif method == "searchlight":
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
    else:  # method == 'roi'
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
    """Auto-default ``standardize=False`` when the user passes a sklearn
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


def resolve_scoring(scoring: str, classifier: bool) -> str:
    """Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor)."""
    if scoring == "auto":
        return "accuracy" if classifier else "r2"
    return scoring


def build_pipeline(model, standardize: bool, reduce: str | None, n_components):
    """Build a per-fold sklearn pipeline: optional StandardScaler → optional
    PCA → model. If only the model is needed, returns the model itself.
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


def _run_whole_brain(bd, X, y, pipe, cv, groups, scoring, refit: bool) -> Predict:
    """Per-fold: fit pipe, score, predict OOF, extract weight_map."""
    from sklearn.base import clone
    from sklearn.metrics import check_scoring

    n_samples = X.shape[0]
    n_voxels = X.shape[1]
    fold_scores: list[float] = []
    fold_predictions = np.zeros(n_samples, dtype=float)
    fold_idx_array = np.full(n_samples, -1, dtype=int)
    fold_weight_maps: list[np.ndarray | None] = []

    scorer = check_scoring(pipe, scoring=scoring)

    for fold_idx, (train_idx, test_idx) in enumerate(_iter_split(cv, X, y, groups)):
        fitted = clone(pipe).fit(X[train_idx], y[train_idx])
        score = scorer(fitted, X[test_idx], y[test_idx])
        fold_scores.append(float(score))
        fold_predictions[test_idx] = fitted.predict(X[test_idx])
        fold_idx_array[test_idx] = fold_idx
        fold_weight_maps.append(_extract_weight_map(fitted, n_voxels))

    scores = np.asarray(fold_scores, dtype=float)
    weight_map_arr, fold_weight_maps_arr = _aggregate_weight_maps(
        fold_weight_maps, n_folds=len(fold_scores), n_voxels=n_voxels
    )

    final_estimator = None
    final_weight_map_arr = None
    if refit:
        final_estimator = clone(pipe).fit(X, y)
        final_weight_map_arr = _extract_weight_map(final_estimator, n_voxels)

    return Predict(
        predictions=fold_predictions,
        scores=scores,
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        cv_folds=fold_idx_array,
        weight_map=_to_braindata(weight_map_arr, bd.mask),
        fold_weight_maps=_to_braindata(fold_weight_maps_arr, bd.mask),
        final_estimator=final_estimator,
        final_weight_map=_to_braindata(final_weight_map_arr, bd.mask),
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


def _extract_weight_map(fitted_pipe, n_voxels: int) -> np.ndarray | None:
    """Extract a (n_voxels,) weight map from the fitted pipeline.

    For multi-class linear classifiers, returns the mean across classes.
    Back-projects through PCA when present. Returns None for non-linear
    models (no .coef_) and emits a one-shot UserWarning.
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
        warnings.warn(
            f"{type(final_est).__name__} has no .coef_ attribute; weight_map "
            "is unavailable for non-linear models. Use a linear model "
            "('svm', 'logistic', 'ridge_classifier', 'lda', 'ridge', 'lasso') "
            "or compute permutation importances directly.",
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

    if coef.shape[0] != n_voxels:
        # Some estimators (e.g., SelectKBest pipelines) reduce feature count
        # in ways we can't trivially reverse — bail out.
        return None
    return coef


def _aggregate_weight_maps(
    per_fold: list[np.ndarray | None], n_folds: int, n_voxels: int
):
    """Stack per-fold weight maps into (n_folds, n_voxels) and mean to
    (n_voxels,). Returns (None, None) if any fold lacked a usable map.
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


def _run_searchlight(
    bd, X, y, pipe, cv, groups, scoring, radius_mm, n_jobs, progress_bar
) -> Predict:
    """Per-voxel-neighborhood CV decoding. Returns Predict with accuracy_map."""
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    from .neighborhoods import compute_searchlight_neighborhoods

    neighborhoods = compute_searchlight_neighborhoods(
        bd.mask, radius_mm=radius_mm, use_cache=True
    )

    def decode_sphere(center_idx, neighbor_indices):
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

    neighborhood_list = list(neighborhoods.iter_neighborhoods())
    if progress_bar:
        try:
            from tqdm import tqdm

            neighborhood_list = list(
                tqdm(
                    neighborhood_list, desc="Searchlight", total=neighborhoods.n_voxels
                )
            )
        except ImportError:
            pass

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


def _run_roi(
    bd, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, progress_bar
) -> Predict:
    """Per-ROI CV decoding. Returns Predict with:

    - ``scores`` ``(n_folds, n_rois)``, ``mean_score`` / ``std_score``
      ``(n_rois,)`` — fold scores per parcel and their cross-fold summary.
    - ``roi_labels`` ``(n_rois,)`` — atlas integer IDs in the same order.
    - ``accuracy_map`` BrainData ``(1, n_voxels)`` — every voxel inside parcel
      *i* set to that parcel's mean accuracy (others NaN).
    """
    from pathlib import Path

    import nibabel as nib
    from joblib import Parallel, delayed
    from nilearn.image import resample_to_img
    from nilearn.masking import apply_mask
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    if roi_mask is None:
        raise ValueError("roi_mask required for method='roi'")

    if isinstance(roi_mask, (str, Path)):
        roi_mask = nib.load(roi_mask)

    if roi_mask.shape != bd.mask.shape or not np.allclose(
        roi_mask.affine, bd.mask.affine
    ):
        roi_mask = resample_to_img(
            roi_mask,
            bd.mask,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    label_vec = apply_mask(roi_mask, bd.mask).astype(np.int64)
    unique_labels = np.unique(label_vec)
    unique_labels = unique_labels[unique_labels != 0]

    n_folds = cv.get_n_splits(X, y, groups=groups)

    def decode_roi(roi_label):
        cols = label_vec == roi_label
        if not cols.any():
            return np.full(n_folds, np.nan, dtype=float)
        X_roi = X[:, cols]
        try:
            scores = cross_val_score(
                clone(pipe), X_roi, y, cv=cv, groups=groups, scoring=scoring
            )
            return np.asarray(scores, dtype=float)
        except Exception:
            return np.full(n_folds, np.nan, dtype=float)

    iterator = unique_labels
    if progress_bar:
        try:
            from tqdm import tqdm

            iterator = tqdm(unique_labels, desc="ROI decoding")
        except ImportError:
            pass

    if n_jobs == 1:
        per_roi_fold_scores = [decode_roi(label) for label in iterator]
    else:
        per_roi_fold_scores = Parallel(n_jobs=n_jobs)(
            delayed(decode_roi)(label) for label in iterator
        )

    # Stack: (n_rois, n_folds) → transpose to (n_folds, n_rois) for consistency
    # with whole-brain `scores` axis-0 = folds.
    fold_scores_per_roi = np.vstack(per_roi_fold_scores).T  # (n_folds, n_rois)
    mean_per_roi = np.nanmean(fold_scores_per_roi, axis=0)  # (n_rois,)
    std_per_roi = np.nanstd(fold_scores_per_roi, axis=0)  # (n_rois,)

    out = np.full(label_vec.shape, np.nan, dtype=float)
    for roi_label, acc in zip(unique_labels, mean_per_roi):
        out[label_vec == roi_label] = acc

    return Predict(
        scores=fold_scores_per_roi,
        mean_score=mean_per_roi,
        std_score=std_per_roi,
        roi_labels=unique_labels.astype(np.int64),
        accuracy_map=_to_braindata(out, bd.mask),
    )
