"""Structural result records returned by decoding operations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .braindata import BrainData


@dataclass(frozen=True)
class Predict:
    """Frozen structural record for `BrainData.predict` decoding results.

    Fields cannot be rebound, but their mutable payloads remain usable. The
    record takes independent ownership of arrays, brain maps, and estimators
    when constructed. Which fields are populated depends on ``spatial_scale``;
    fields not applicable to the call stay ``None`` and are dropped by
    `available` and `asdict`.

    **Brain-space outputs are `BrainData` objects**, not raw arrays, so
    ``result.weight_map.plot()`` works directly (``.data`` gives the array).
    Non-spatial fields (``predictions``, ``cv_folds``, scalar scores) are
    numpy.

    **Populated by `spatial_scale`.** ``'whole_brain'``: ``predictions``,
    ``scores``, ``mean_score``, ``std_score``, ``cv_folds``, ``weight_map``,
    ``fold_weight_maps``, ``estimator``. ``'roi'``: ``scores``,
    ``mean_score``, ``std_score``, ``roi_labels``, ``accuracy_map``,
    ``weight_map``, ``fold_weight_maps``, ``estimator`` — and if any parcel's
    model cannot expose ``coef_`` (a non-linear model, or feature selection in
    the pipeline), ``weight_map`` / ``fold_weight_maps`` / ``estimator`` are
    all ``None`` for the whole call. ``'searchlight'``: ``accuracy_map`` only.

    **Why the all-data fit is the canonical map.** The mean of per-fold
    ``coef_`` vectors corresponds to no actual fitted estimator (each fold saw
    a different subset). The all-data refit is one real model using all the
    information; CV gives the honest *score*, the refit gives the publishable
    *map*. ``fold_weight_maps`` is still exposed for stability analysis, and
    the CV mean is ``fold_weight_maps.data.mean(axis=0)``.

    Attributes:
        predictions (ndarray | None): Out-of-fold CV predictions,
            ``(n_samples,)`` (whole-brain only).
        scores (ndarray | None): Per-fold score — ``(n_folds,)`` for
            whole-brain, ``(n_folds, n_rois)`` for ROI.
        mean_score (float | ndarray | None): Mean score across folds — a float
            for whole-brain, ``(n_rois,)`` for ROI.
        std_score (float | ndarray | None): Score standard deviation across
            folds, same form as ``mean_score``.
        cv_folds (ndarray | None): Fold index per sample, ``(n_samples,)``
            (whole-brain only).
        roi_labels (ndarray | None): Atlas integer ids, ``(n_rois,)``, in the
            order of ``mean_score`` / ``std_score`` / ``scores`` axis 1 (ROI
            only).
        accuracy_map (BrainData | None): ``(1, n_voxels)`` map — for ROI,
            every voxel in parcel *i* holds that parcel's mean score (NaN
            outside parcels); for searchlight, the sphere-centered score at
            each voxel.
        weight_map (BrainData | None): ``(1, n_voxels)`` ``coef_`` of the model
            refit on all data — the publishable map. For ROI, each parcel's
            coefficients are written back into voxel space (NaN outside
            parcels); magnitudes are not comparable across parcels.
        fold_weight_maps (BrainData | None): ``(n_folds, n_voxels)`` stack of
            per-fold ``coef_`` for stability analysis.
        estimator (Any): The fitted all-data sklearn estimator (whole-brain;
            use it to ``.predict()`` on new data), or a ``dict[int, estimator]``
            keyed by atlas label (ROI).

    Note:
        Encoding-model timeseries prediction (``bd.predict(X=...)``) returns a
        `BrainData` directly rather than a `Predict` — the natural container
        for a voxel timeseries.
    """

    # MVPA / classification — populated when y given
    predictions: np.ndarray | None = None
    scores: np.ndarray | None = None
    mean_score: float | np.ndarray | None = None
    std_score: float | np.ndarray | None = None
    cv_folds: np.ndarray | None = None

    # ROI — atlas labels in the order of mean_score / std_score / scores axis 1
    roi_labels: np.ndarray | None = None

    # Brain-space maps — BrainData when populated by the runners
    accuracy_map: BrainData | None = None
    weight_map: BrainData | None = None
    fold_weight_maps: BrainData | None = None

    # All-data fitted estimator (whole_brain only)
    estimator: Any = None

    def __post_init__(self):
        """Take independent ownership of every populated mutable payload."""
        from .braindata import BrainData

        for name in ("accuracy_map", "weight_map", "fold_weight_maps"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BrainData):
                raise TypeError(
                    f"{name} is {type(value).__name__}, expected BrainData."
                )
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, deepcopy(value))

    def available(self) -> list:
        """Return names of non-None fields (excludes private)."""
        return [
            field_name
            for field_name in self.__dataclass_fields__
            if not field_name.startswith("_") and getattr(self, field_name) is not None
        ]

    def asdict(self, include_none: bool = False) -> dict:
        """Convert to dictionary.

        Args:
            include_none: If True, include fields with None values.
                Private fields (starting with _) are always excluded.

        Returns:
            Dictionary of field names to values.
        """
        full_dict = dataclass_asdict(self)
        filtered = {k: v for k, v in full_dict.items() if not k.startswith("_")}
        if not include_none:
            filtered = {k: v for k, v in filtered.items() if v is not None}
        return filtered
