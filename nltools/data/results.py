"""Structural result records returned by decoding operations."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import asdict as dataclass_asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

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
            keyed by atlas label (ROI). ``None`` when read back from a cached
            bundle.
        permutation_scores (ndarray | None): Label-permutation null from
            `BrainCollection.predict_group` — ``(n_permute,)`` for whole-brain,
            ``(n_permute, n_rois)`` for ROI, ``(n_permute, n_voxels)`` for
            searchlight.
        permutation_pvalue (float | ndarray | BrainData | None): Upper-tail
            permutation p-value — a float, ``(n_rois,)``, or a
            ``(1, n_voxels)`` `BrainData` map, matching ``permutation_scores``.

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

    # Label-permutation null (BrainCollection.predict_group(n_permute=)).
    # Shapes follow spatial_scale: whole_brain → scores (n_permute,) with a
    # float p; roi → scores (n_permute, n_rois) with p (n_rois,); searchlight
    # → scores (n_permute, n_voxels) with p as a BrainData (1, n_voxels) map.
    permutation_scores: np.ndarray | None = None
    permutation_pvalue: float | np.ndarray | BrainData | None = None

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


@dataclass(frozen=True)
class PredictCollection:
    """Frozen structural record for per-subject decoding results.

    Returned by ``BrainCollection.predict(y=...)``: one `Predict` per subject
    (each an independent within-subject model), plus the collection's
    per-subject metadata. Sequence-like — ``len``, iteration, and integer
    indexing all address the underlying `Predict` objects.

    The stacking properties are the bridge to second-level inference: the
    per-subject maps become one ``BrainData (n_subjects, n_voxels)``, ready
    for a group test.

    Attributes:
        results (tuple[Predict, ...]): One `Predict` per subject, in
            collection order.
        metadata (pl.DataFrame | None): Per-subject metadata (one row per
            subject), carried over from the source collection.
        paths (tuple[Path | None, ...] | None): On-disk predict-bundle paths,
            populated when the producing call cached its results
            (``cache=True`` / ``'auto'``).
        mean_scores (np.ndarray): Stacked per-subject mean CV score —
            ``(n_subjects,)`` for whole-brain decoding, ``(n_subjects,
            n_rois)`` for ROI.
        std_scores (np.ndarray): Stacked per-subject score standard deviation
            across folds, same shape as ``mean_scores``.
        scores (pl.DataFrame): Per-subject score table — the metadata plus
            ``mean_score`` / ``std_score`` columns. Whole-brain decoding only;
            ROI results raise (use ``mean_scores`` / ``std_scores``).
        weight_maps (BrainData): Per-subject decoder maps stacked into one
            ``(n_subjects, n_voxels)`` `BrainData`.
        accuracy_maps (BrainData): Per-subject accuracy maps stacked into one
            ``(n_subjects, n_voxels)`` `BrainData`.

    Examples:
        ```python
        pc = collection.predict(y="condition", cv=5)
        pc.scores                    # per-subject accuracy table
        pc[0].weight_map.plot()      # one subject's decoder map

        # Second-level inference on the decoder maps:
        from nltools.algorithms import one_sample_permutation_test
        group = one_sample_permutation_test(pc.weight_maps.data)
        ```
    """

    results: tuple[Predict, ...]
    metadata: pl.DataFrame | None = None
    paths: tuple[Path | None, ...] | None = None

    def __post_init__(self):
        results = tuple(deepcopy(self.results))
        if not results:
            raise ValueError("PredictCollection cannot be empty.")
        for i, r in enumerate(results):
            if not isinstance(r, Predict):
                raise TypeError(
                    f"results[{i}] is {type(r).__name__}, expected Predict."
                )
        object.__setattr__(self, "results", results)
        if self.metadata is not None:
            object.__setattr__(self, "metadata", deepcopy(self.metadata))
        if self.metadata is not None and self.metadata.shape[0] != len(results):
            raise ValueError(
                f"metadata has {self.metadata.shape[0]} rows for "
                f"{len(results)} results."
            )
        if self.paths is not None:
            paths = tuple(self.paths)
            if len(paths) != len(results):
                raise ValueError(
                    f"paths has {len(paths)} entries for {len(results)} results."
                )
            object.__setattr__(self, "paths", deepcopy(paths))

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[Predict]:
        return iter(self.results)

    def __getitem__(self, idx: int) -> Predict:
        return self.results[idx]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_subjects={len(self.results)}, "
            f"available={self.available()})"
        )

    def _stack_field(self, field: str) -> np.ndarray:
        values = [getattr(r, field) for r in self.results]
        missing = [i for i, v in enumerate(values) if v is None]
        if missing:
            raise ValueError(
                f"{field} is not populated for subjects {missing} — it is "
                f"unavailable for their dispatch mode (see `Predict`)."
            )
        return np.stack([np.asarray(v) for v in values], axis=0)

    @property
    def mean_scores(self) -> np.ndarray:
        """Per-subject mean CV score — ``(n_subjects,)`` or ``(n_subjects, n_rois)``."""
        return self._stack_field("mean_score")

    @property
    def std_scores(self) -> np.ndarray:
        """Per-subject score std across folds, matching ``mean_scores``' shape."""
        return self._stack_field("std_score")

    @property
    def scores(self):
        """Per-subject score table (polars): metadata + mean/std score columns.

        Only defined when each subject's score summary is a scalar
        (whole-brain decoding); for ROI decoding use ``mean_scores`` /
        ``std_scores`` — the array forms.
        """
        import polars as pl

        mean = self.mean_scores
        if mean.ndim != 1:
            raise ValueError(
                "scores is a per-subject scalar table; these results carry "
                "array-valued score summaries (ROI decoding) — use "
                "mean_scores / std_scores instead."
            )
        base = (
            self.metadata
            if self.metadata is not None
            else pl.DataFrame({"subject": np.arange(len(self))})
        )
        return base.with_columns(
            pl.Series("mean_score", mean),
            pl.Series("std_score", self._stack_field("std_score")),
        )

    def _stack_maps(self, field: str):
        maps = [getattr(r, field) for r in self.results]
        missing = [i for i, m in enumerate(maps) if m is None]
        if missing:
            raise ValueError(
                f"{field} is not populated for subjects {missing} — it is "
                f"unavailable for their dispatch mode (see `Predict`)."
            )
        from nltools.data import BrainData

        stacked = np.vstack([np.asarray(m.data).reshape(1, -1) for m in maps])
        return BrainData(stacked, mask=maps[0].mask)

    @property
    def weight_maps(self):
        """Per-subject decoder maps as one ``BrainData (n_subjects, n_voxels)``."""
        return self._stack_maps("weight_map")

    @property
    def accuracy_maps(self):
        """Per-subject accuracy maps as one ``BrainData (n_subjects, n_voxels)``."""
        return self._stack_maps("accuracy_map")

    def available(self) -> list:
        """Return field names populated on every subject's `Predict`."""
        common = set(self.results[0].available())
        for r in self.results[1:]:
            common &= set(r.available())
        return sorted(common)
