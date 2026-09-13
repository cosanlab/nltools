"""Coefficient back-projection for MVPA decoding pipelines.

A decoding pipeline preprocesses voxels before it fits, so the coefficients it
learns live on the preprocessed feature axis, not on the voxel axis a brain map
needs. These functions walk a *fitted* scikit-learn estimator or `Pipeline`
backwards and return coefficients on the original voxel axis.

Everything here is a pure function over fitted scikit-learn objects: no
`BrainData`, no file I/O, no global state. `nltools.data.braindata.prediction`
orchestrates; this module does the numerics.

Only the preprocessing steps in `SUPPORTED_TRANSFORMERS` are accepted. A
transformer outside that set is rejected even when it implements
`inverse_transform`, because inverting a *data* transformation is not the same
operation as back-projecting a *coefficient* vector: `Normalizer`, for
instance, rescales each observation rather than each feature, so its
coefficients have no fixed voxel-space image.

Centering is deliberately not undone. It shifts the intercept of the raw-space
decision function without changing its slope map, so `raw_data @ weight_map`
need not reproduce the decision function. Use the fitted estimator itself to
predict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    RFECV,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectorMixin,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Preprocessing steps a decoding pipeline may contain. Each one has a defined
#: coefficient back-projection: a scaler rescales weights, `PCA` rotates them,
#: and every selector expands them with exact zeros at the positions it dropped.
SUPPORTED_TRANSFORMERS: tuple[type, ...] = (
    StandardScaler,
    PCA,
    VarianceThreshold,
    GenericUnivariateSelect,
    SelectPercentile,
    SelectKBest,
    SelectFpr,
    SelectFdr,
    SelectFwe,
    SelectFromModel,
    RFE,
    RFECV,
    SequentialFeatureSelector,
)


class _BackProjectionError(ValueError):
    """A fitted pipeline's coefficients cannot be projected onto the voxel axis.

    A `ValueError` subclass, so callers that catch `ValueError` — including the
    public `BrainData.predict` contract — see it as one, while the internal
    runners can still tell it apart from an unrelated fit failure.
    """


def _is_passthrough(step: Any) -> bool:
    """Whether a pipeline step is a placeholder that transforms nothing."""
    return step is None or (isinstance(step, str) and step == "passthrough")


def _split_pipeline(pipeline: Any) -> tuple[list, Any]:
    """Split a decoding pipeline into its preprocessing steps and final estimator.

    Args:
        pipeline: A `Pipeline`, or a bare estimator (a pipeline of one step).

    Returns:
        tuple: ``(steps, final_estimator)``. A bare estimator has no steps.
    """
    if isinstance(pipeline, Pipeline):
        steps = [step for _, step in pipeline.steps]
        return steps[:-1], steps[-1]
    return [], pipeline


def _validate_decoding_pipeline(pipeline: Any) -> None:
    """Check a pipeline's structure before anything is fitted.

    Catches the two failures that are visible without fitting: a preprocessing
    step outside `SUPPORTED_TRANSFORMERS`, and a `OneVsRestClassifier` that is
    not the final step. Whether the final estimator exposes ``coef_`` can only
    be observed after a fit, so `_back_project_weight_maps` checks that.

    Args:
        pipeline: The estimator or `Pipeline` MVPA is about to fit.

    Raises:
        _BackProjectionError: If a step is unsupported or misplaced.
    """
    steps, _ = _split_pipeline(pipeline)
    for step in steps:
        if _is_passthrough(step):
            continue
        if isinstance(step, OneVsRestClassifier):
            raise _BackProjectionError(_ovr_not_final_message())
        if not isinstance(step, SUPPORTED_TRANSFORMERS):
            raise _BackProjectionError(_unsupported_step_message(step))


def _whitening_scale(explained_variance: np.ndarray) -> np.ndarray:
    """Return the per-component scale a whitened `PCA` divides its output by.

    ``sqrt(explained_variance_)``, with values below the dtype's epsilon
    replaced by that epsilon so a degenerate component cannot blow the
    back-projected weights up to infinity.

    Args:
        explained_variance: The fitted ``PCA.explained_variance_`` vector.

    Returns:
        ndarray: The floored component scales, one per component.
    """
    scale = np.sqrt(np.asarray(explained_variance, dtype=float))
    eps = np.finfo(scale.dtype).eps
    return np.where(scale < eps, eps, scale)


def _coefficient_rows(final_estimator: Any) -> np.ndarray:
    """Return a fitted estimator's coefficients as ``(n_maps, n_final_features)``.

    One row for a regressor or a binary classifier — the signed map for
    ``classes_[1]`` versus ``classes_[0]`` — and one row per class, in
    ``classes_`` order, for a multiclass classifier. Rows are never averaged: a
    mean across classes describes no fitted decision boundary.

    `OneVsRestClassifier` is handled explicitly because it exposes no combined
    ``coef_``; its fitted children are read in class order instead.

    Args:
        final_estimator: The fitted estimator ending the pipeline.

    Returns:
        ndarray: Coefficients, ``(n_maps, n_final_features)``.

    Raises:
        _BackProjectionError: If the estimator exposes no usable ``coef_``.
    """
    if isinstance(final_estimator, OneVsRestClassifier):
        return _one_vs_rest_rows(final_estimator)
    coef = _coef_of(final_estimator)
    if coef.ndim != 2:
        raise _BackProjectionError(
            f"{type(final_estimator).__name__}.coef_ has shape {coef.shape}; "
            f"a decoding estimator must expose one coefficient row per map."
        )
    return coef


def _back_project_step(weights: np.ndarray, step: Any) -> np.ndarray:
    """Project coefficients backwards through one fitted preprocessing step.

    Args:
        weights: Coefficients on the step's *output* feature axis,
            ``(n_maps, n_output_features)``.
        step: The fitted transformer, or a passthrough placeholder.

    Returns:
        ndarray: Coefficients on the step's *input* feature axis,
            ``(n_maps, n_input_features)``.

    Raises:
        _BackProjectionError: If the step is unsupported, or its fitted output
            width does not match the incoming coefficients.
    """
    if _is_passthrough(step):
        return weights
    if not isinstance(step, SUPPORTED_TRANSFORMERS):
        # The same whitelist `_validate_decoding_pipeline` applies before
        # fitting, re-checked here so a direct caller cannot skip it.
        raise _BackProjectionError(_unsupported_step_message(step))
    if isinstance(step, StandardScaler):
        _check_width(weights, int(step.n_features_in_), step)
        if step.with_std and step.scale_ is not None:
            return weights / step.scale_
        return weights
    if isinstance(step, PCA):
        components = step.components_
        _check_width(weights, components.shape[0], step)
        if step.whiten:
            weights = weights / _whitening_scale(step.explained_variance_)
        return weights @ components
    if isinstance(step, SelectorMixin):
        support = step.get_support()
        _check_width(weights, int(support.sum()), step)
        expanded = np.zeros((weights.shape[0], support.size), dtype=weights.dtype)
        expanded[:, support] = weights
        return expanded
    # Unreachable today: every whitelisted class is a StandardScaler, a PCA, or
    # a SelectorMixin. It guards a future whitelist entry that forgets a branch.
    raise _BackProjectionError(_unsupported_step_message(step))  # pragma: no cover


def _back_project_weight_maps(fitted_estimator: Any, n_features: int) -> np.ndarray:
    """Project a fitted pipeline's coefficients onto the original feature axis.

    Starts from ``(n_maps, n_final_features)`` coefficients and walks the fitted
    preprocessing steps in reverse order, validating each step's widths, until
    the weights sit on the axis the pipeline was fitted from — the whole-brain,
    parcel, or sphere voxel axis, depending on the caller.

    Args:
        fitted_estimator: A fitted estimator or `Pipeline`.
        n_features: Width of the original feature axis the maps must land on.

    Returns:
        ndarray: ``(n_maps, n_features)`` — one row for regression and binary
            classification, one row per class for multiclass.

    Raises:
        _BackProjectionError: If the final estimator exposes no ``coef_``, a step
            is unsupported or misplaced, or any width does not line up.

    Examples:
        ```python
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        pipe = make_pipeline(StandardScaler(), LinearSVC()).fit(X, y)
        maps = _back_project_weight_maps(pipe, X.shape[1])
        # → (1, n_voxels), in raw voxel units
        ```
    """
    steps, final_estimator = _split_pipeline(fitted_estimator)
    for step in steps:
        if isinstance(step, OneVsRestClassifier):
            raise _BackProjectionError(_ovr_not_final_message())
    weights = _coefficient_rows(final_estimator)
    for step in reversed(steps):
        weights = _back_project_step(weights, step)
    if weights.shape[1] != n_features:
        raise _BackProjectionError(
            f"Back-projected coefficients have width {weights.shape[1]}, but "
            f"the original feature axis has width {n_features}. The pipeline's "
            f"fitted steps do not reach back to the voxels it was fitted from."
        )
    return weights


def _coef_of(estimator: Any) -> np.ndarray:
    """Return an estimator's ``coef_`` as a 2-D array, or raise if it has none."""
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        raise _BackProjectionError(
            f"{type(estimator).__name__} exposes no coef_, so this decoding "
            f"pipeline produces no weight map. Use a linear estimator — "
            f"'linear_svc', 'logistic_regression', "
            f"'linear_discriminant_analysis', 'ridge_classifier', 'ridge', "
            f"'lasso', 'linear_svr', or any sklearn estimator with coef_."
        )
    coef = np.asarray(coef, dtype=float)
    return coef[None, :] if coef.ndim == 1 else coef


def _one_vs_rest_rows(ovr: OneVsRestClassifier) -> np.ndarray:
    """Stack a fitted `OneVsRestClassifier`'s child coefficient rows in class order."""
    classes = np.asarray(ovr.classes_)
    children = list(ovr.estimators_)
    expected = 1 if len(classes) == 2 else len(classes)
    if len(children) != expected:
        raise _BackProjectionError(
            f"OneVsRestClassifier fitted {len(children)} child estimator(s) for "
            f"{len(classes)} classes; {expected} were expected."
        )
    rows = []
    for child in children:
        coef = _coef_of(child)
        if coef.shape[0] != 1:
            raise _BackProjectionError(
                f"Each OneVsRestClassifier child must expose one coefficient "
                f"row; {type(child).__name__} exposes {coef.shape[0]}."
            )
        rows.append(coef[0])
    return np.vstack(rows)


def _check_width(weights: np.ndarray, expected: int, step: Any) -> None:
    """Require incoming coefficients to match a fitted step's output width."""
    if weights.shape[1] != expected:
        raise _BackProjectionError(
            f"{type(step).__name__} was fitted to produce {expected} features, "
            f"but the coefficients arriving at it have width "
            f"{weights.shape[1]}. The pipeline's fitted steps do not line up."
        )


def _ovr_not_final_message() -> str:
    """Explain why `OneVsRestClassifier` cannot sit inside a pipeline."""
    return (
        "OneVsRestClassifier must be the final pipeline step: it holds one "
        "fitted child per class and exposes no combined coef_, so nothing "
        "downstream of it can be back-projected. Put the shared preprocessing "
        "before it."
    )


def _unsupported_step_message(step: Any) -> str:
    """Explain why a preprocessing step has no coefficient back-projection."""
    supported = ", ".join(cls.__name__ for cls in SUPPORTED_TRANSFORMERS)
    return (
        f"{type(step).__name__} is not a supported decoding preprocessing step, "
        f"even if it implements inverse_transform: inverting a data "
        f"transformation is not the same as back-projecting coefficients. "
        f"Supported steps are {supported}, plus None and 'passthrough'."
    )
